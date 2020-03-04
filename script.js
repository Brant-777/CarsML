/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 * This is an asynchronous where lower parts are lifted into the code in sequence 1 before the other
 * carsData is lifted into carsDataReq to await the function before 
 * filter removes nulls because nulls are not treated as zero and returns cleaned json file removing nulls and only including mpg and Displacement
 */
async function getData() {
  const carsDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');  
  const carsData = await carsDataReq.json();  
  const cleaned = carsData.map(car => ({
    mpg: car.Miles_per_Gallon,
    Displacement: car.Displacement,
  }))
  .filter(car => (car.mpg != null && car.Displacement != null));
  
  return cleaned;
}




//use DOM Chart(tfjs-vis) to see if you have any data that strays to far from the normal or can be ignored (similar patterns)
document.addEventListener('DOMContentLoaded', run);

//console.log(data);

/*
middle of neural network weights are placed
*/
function createModel() {
  // Create a sequential model
  const model = tf.sequential(); 
  
  // Add a single hidden layer
  /*This adds a hidden layer to our network. A dense layer is a type of layer that multiplies its inputs by a matrix (called weights) and then adds a number (called the bias) to the result. As this is the first layer of the network, we need to define our inputShape. The inputShape is [1] because we have 1 number as our input (the Displacement of a given car).
   units sets how big the weight matrix will be in the layer. By setting it to 1 here we are saying there will be 1 weight for each of the input features of the data.
   Hidden Layer 1
   */
  
  model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
  

  /*
  Hidden Layer Sigmoid
  */


  // Add an output layer
  /*
  Dense layers come with a bias term by default, so we do not need to set useBias to true, we will omit from further calls to tf.layers.dense
  -Output Layer-
  */
  model.add(tf.layers.dense({units: 1, useBias: true}));

  return model;
}

// Create the model
const model = createModel(); 

/**
 * Convert the input data to tensors that we can use for machine 
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 * Shuffle the data so its not dependent on order or sensitive to subject subgroups
 */
function convertToTensor(data) {
  // Wrapping these calculations in a tidy will dispose any 
  // intermediate tensors.
  
  return tf.tidy(() => {
    // Step 1. Shuffle the data    
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor 2d tensor/2d object
    const inputs = data.map(d => d.Displacement)
    const labels = data.map(d => d.mpg);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    /*
    Here we normalize the data into the numerical range 0-1 using min-max scaling. Normalization is important because the internals of many machine learning models you will build with tensorflow.js are designed to work with numbers that are not too big. Common ranges to normalize data to include 0 to 1 or -1 to 1. You will have more success training your models if you get into the habit of normalizing your data to some reasonable range.
    */
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();  
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later. (used to un normalize output)
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  });  
}

async function trainModel(model, inputs, labels) {
  // Prepare the model for training.
  /*
  optimizer: This is the algorithm that is going to govern the updates to the model as it sees examples. There are many optimizers available in TensorFlow.js. The adam optimizer as it is quite effective in practice and requires no configuration.
loss: this is a function that will tell the model how well it is doing on learning each of the batches (data subsets) that it is shown. Here we use meanSquaredError to compare the predictions made by the model with the true values.
  */  
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });
  

  /*
batchSize refers to the size of the data subsets that the model will see on each iteration of training. Common batch sizes tend to be in the range 32-512. There isn't really an ideal batch size for all problems and it is beyond the scope of this tutorial to describe the mathematical motivations for various batch sizes.
epochs refers to the number of times the model is going to look at the entire dataset that you provide it. Here we will take 50 iterations through the dataset.
  */
  const batchSize = 32;
  const epochs = 50;
  
/*
To monitor training progress we pass some callbacks to model.fit. We use tfvis.show.fitCallbacks to generate functions that plot charts for the ‘loss' and ‘mse' metric we specified earlier.
* Learning rate: how much you update the weights in the direction to decrease the gradient (based on adam weights)
*Epoch: iteration of going through the whole dataset in this case const epochs = 50
*Least Mean Square Error: or LMSE is the distance between the predicted output Yp and the true output Y given by LMSE = (Yp - Y)^2 
*/

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'], 
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}



function testModel(model, inputData, normalizationData) {
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;  
  
  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling 
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {
    
    /*
    generate 100 new ‘examples' to feed to the model. Model.predict is how we feed those examples into the model. Note that they need to be have a similar shape ([num_examples, num_features_per_example])
    */
    const xs = tf.linspace(0, 1, 100);      
    const preds = model.predict(xs.reshape([100, 1]));      
    
    /*
    get the data back to our original range (rather than 0-1) invert operations
    */
    const unNormXs = xs
      .mul(inputMax.sub(inputMin))
      .add(inputMin);
    
    const unNormPreds = preds
      .mul(labelMax.sub(labelMin))
      .add(labelMin);
    
    /* Un-normalize the data
    .dataSync() is a method we can use to get a typedarray of the values stored in a tensor. This allows us to process those values in regular JavaScript. This is a synchronous version of the .data() method which is generally preferred.
    */
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });
  
 
  const predictedPoints = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  });
  
  const originalPoints = inputData.map(d => ({
    x: d.Displacement, y: d.mpg,
  }));
  
  
  tfvis.render.scatterplot(
    {name: 'Model Predictions vs Original Data'}, 
    {values: [originalPoints, predictedPoints], series: ['original', 'predicted']}, 
    {
      xLabel: 'Displacement',
      yLabel: 'MPG',
      height: 300
    }
  );
}

async function run() {
  /** Load and plot the original input data that we are going to train on.
  * console log for getData funciton not sure why it returns outside of async run?
  * Our goal is to train a model that will take one number, Displacement and learn to predict one number, Miles per Gallon. Remember that one-to-one mapping(supervised learning)
  */
  const data = await getData();
  //console.log(data);
  const values = data.map(d => ({
    x: d.Displacement,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    {name: 'Displacement v MPG'},
    {values}, 
    {
      xLabel: 'Displacement',
      yLabel: 'MPG',
      height: 300
    }
  );


  // More code will be added below
  
    tfvis.show.modelSummary({name: 'Model Summary'}, model);

    

  // Convert the data to a form we can use for training.
    const tensorData = convertToTensor(data);
    const {inputs, labels} = tensorData;

    
  // Train the model  
   await trainModel(model, inputs, labels);
   console.log('Done Training');  

   // Make some predictions using the model and compare them to the
   // original data
   //when training and testing a model the predicted result may be different each time the code is run so it may require the model to be used more than once to determine a regression of the right fit
    testModel(model, data, tensorData);

}

//prompt(getData);
//console.log("Hello TensorFlow");
//console.log(tfvis)
