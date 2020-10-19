require('@tensorflow/tfjs-node');

const _ = require('async-dash');
const paths = require('path');
const canvas = require('canvas');
const faceapi = require('face-api.js');

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

let isPrepared = false;

exports.faceApiOptions = null;

exports.prepareFaceApi = async (options = {}) => {
  if (this.isPrepared) return;

  const vOptions = {
    faceDetectionNet: faceapi.nets.ssdMobilenetv1,
    minConfidence: 0.5,
    ...options,
  };
  const weightsPath = paths.resolve(__dirname, './weights');

  await vOptions.faceDetectionNet.loadFromDisk(weightsPath);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(weightsPath);
  await faceapi.nets.ageGenderNet.loadFromDisk(weightsPath);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(weightsPath);

  this.faceApiOptions = new faceapi.SsdMobilenetv1Options({
    minConfidence: vOptions.minConfidence,
  });

  this.isPrepared = true;
};

exports.getNets = async () => {
  return _.keys(faceapi.nets);
};

const translateGender = (gender) => _.get({
  male: '男',
  female: '女',
}, gender, '未知');
exports.translateGender = translateGender;

exports.detectFaces = async (imagePath) => {
  await this.prepareFaceApi();

  if (!imagePath) {
    throw new Error('Please input imagePath');
  }

  const img = await canvas.loadImage(imagePath);
  const results = await faceapi
    .detectAllFaces(img, this.faceApiOptions)
    .withFaceLandmarks()
    .withAgeAndGender()
    .withFaceDescriptors();

  const out = faceapi.createCanvasFromMedia(img);
  faceapi.draw.drawDetections(
    out,
    _.map(results, (res, index) => `#${index + 1} ${res.detection}`),
  );
  results.forEach((result) => {
    const { age, gender, genderProbability } = result;
    new faceapi.draw.DrawTextField(
      [
        `${faceapi.utils.round(age, 0)} years`,
        `${translateGender(gender)} (${faceapi.utils.round(genderProbability) * 100}%)`,
      ],
      result.detection.box.bottomLeft,
    ).draw(out);
  });

  const getJson = () => {
    return _.map(results, (item) => ({
      ..._.pick(item, ['gender', 'genderProbability', 'age']),
    }));
  };

  const getJpgBuffer = () => {
    return out.toBuffer('image/jpeg');
  };

  return {
    raw: results,
    res: getJson(),
    getJpgBuffer,
  };
};

exports.buildDescriptor = async (faces = []) => {
  const ret = [];
  await _.asyncEach(faces, async (face) => {
    const { label, imagePaths, descriptors } = face;
    const vDescriptors = [];
    if (imagePaths && imagePaths.length) {
      await _.asyncEach(imagePaths, async (imagePath) => {
        const inputImage = await canvas.loadImage(imagePath);
        const imageRes = await faceapi
          .detectSingleFace(inputImage)
          .withFaceLandmarks()
          .withAgeAndGender()
          .withFaceDescriptor();
          vDescriptors.push(imageRes.descriptor);
      });
    }
    if (descriptors && descriptors.length) {
      vDescriptors.push(...descriptors);
    }
    ret.push(new faceapi.LabeledFaceDescriptors(label, vDescriptors));
  });
  return ret;
};

exports.recognizeFaces = async (descriptor, queryImagePath) => {
  await this.prepareFaceApi();

  if (!queryImagePath) throw new Error('Please input queryImage');

  const queryImage = await canvas.loadImage(queryImagePath);

  const resultsQuery = await faceapi
    .detectAllFaces(queryImage, this.faceApiOptions)
    .withFaceLandmarks()
    .withAgeAndGender()
    .withFaceDescriptors();

  const faceMatcher = new faceapi.FaceMatcher(descriptor);

  const labels = faceMatcher.labeledDescriptors.map((ld) => ld.label);

  const resRecognition = resultsQuery.map((res) => {
    const bestMatch = faceMatcher.findBestMatch(res.descriptor); 
    const ret = {
      ..._.pick(res, ['gender', 'genderProbability', 'age']),
      bestMatch,
      matchLabel: bestMatch.label,
      matchDistance: bestMatch.distance,
    };
    if (bestMatch.distance > 0.5) {
      _.extend(ret, {
        bestMatch,
        matchLabel: '',
        matchDistance: 1,
      });
    }
    return {
      ...res,
      ...ret,
    };
  });

  const outQuery = faceapi.createCanvasFromMedia(queryImage);
  resRecognition.forEach((res) => {
    new faceapi.draw.DrawBox(res.detection.box, {
      label: res.matchDistance < 0.5
        ? `${res.matchLabel} (${((1 - res.matchDistance) * 100).toFixed(2)}%)`
        : '未匹配',
    }).draw(outQuery);
    new faceapi.draw.DrawTextField(
      [
        `${faceapi.utils.round(res.age, 0)} years`,
        `${translateGender(res.gender)} (${faceapi.utils.round(res.genderProbability) * 100}%)`,
      ],
      res.detection.box.bottomLeft,
    ).draw(outQuery);
  });

  const getQueryJpgBuffer = () => {
    return outQuery.toBuffer('image/jpeg');
  };

  const getJson = () => {
    return _.map(resRecognition, (item) => ({
      label: item.matchLabel,
      bestMatchDistance: item.matchDistance,
    }));
  };

  return {
    raw: resRecognition,
    res: getJson(),
    getQueryJpgBuffer,
  };
};
