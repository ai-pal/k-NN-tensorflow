# Learn K-NN with TensorFlow JS :sunglasses:

This repository focuses on developing a interactive ground to understand K-NN algorithm with the :heart: of `TensorFlow JS`.

For this mission we will try to make a model that predicts Real Estate Property prices. 

## How we Progress

Initially we will use a training data set 21 612 records in CSV format.

Then we will write our own CSV loader function to load the data.

Finally we will use tensors to predict house values using knn.

## K-NN Algorithm using TF

```js
function knn(features, labels, predictionPoint, k) {
    const { mean, variance } = tf.moments(features, 0);

    const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5))

    return features
        .sub(mean)
        .div(variance.pow(0.5))
        .sub(scaledPrediction)
        .pow(2)
        .sum(1)
        .pow(0.5)
        .expandDims(1)
        .concat(labels, 1)
        .unstack()
        .sort((a, b) => a.get(0) > b.get(0) ? 1 : -1)
        .slice(0, k)
        .reduce((acc, pair) => acc + pair.get(1), 0) / k
}
```