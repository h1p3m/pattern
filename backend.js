async function loadModuleFromUrl(url) {
  const response = await fetch(url);
  const moduleString = await response.text();
  const module = eval(moduleString);
  return module;
}

const tf = require('./tfjs');

const keras = require('./keras');




const express = require('express');
const app = express();
const port = 3000;
//file system
const fs = require('fs');


//cors
const cors = require('cors');
app.use(cors());

//body parser
const bodyParser = require('body-parser');
app.use(bodyParser.json());



// const tf = require('@tensorflow/tfjs');
// require('@tensorflow/tfjs-node');

//define exec
const exec = require('child_process').exec;


// exec('wget https://cdn.jsdelivr.net/npm/@tensorflow/tfjs', function(stdout) {  });

const { type } = require('os');
//get maximum call stack size exceeded
const process=require('process');

const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');

//const { performance } = require('perf_hooks');

const v8 = require('v8');
let maxcsse = 0;


class  MindPattern { //mind control
    constructor() {
      this.model = [];

      this.tasksQueue = []; //name of task, params and priority


      this.temp=[];

      this.short=[];
      this.lastBrainwave = 1; //need harmonical all temp weights and biases
      this.isTraining = false;

      this.isPredicting = false;

      this.isEvaluating = false;

      this.isSaving = false;

      this.isLoading = false;

      this.outputs= [];
    }
  

   geneticModelConstructor(numGenes) {
      // Define the gene pool
      const genePool = ['A', 'C', 'G', 'T'];
    
      // Create the initial model with random genes
      const model = [];
      for (let i = 0; i < numGenes; i++) {
        const randomIndex = Math.floor(Math.random() * genePool.length);
        model.push(genePool[randomIndex]);
      }
    
      return model;
    }
    

    drawMatrix(matrix) {
    const numRows = matrix.length;
    const numCols = matrix[0].length;
    const totalCells = numRows * numCols;
    const result = [];
  
    let i = 0, j = 0, di = 0, dj = 1;
    let minI = 0, minJ = 0, maxI = numRows - 1, maxJ = numCols - 1;
  
    for (let k = 0; k < totalCells; k++) {
      result.push(matrix[i][j]);
      const arrow = getArrowDirection(di, dj);
      console.log(`[${i}][${j}] ${arrow}`);
      
      if (i === minI && j === minJ) {
        di = dj;
        dj = 0;
        minI++;
      } else if (i === maxI && j === maxJ) {
        di = -dj;
        dj = 0;
        maxI--;
      } else if (i === minI && j === maxJ) {
        const temp = di;
        di = 0;
        dj = -temp;
        minJ++;
      } else if (i + di < minI || i + di > maxI || j + dj < minJ || j + dj > maxJ) {
        const temp = di;
        di = dj;
        dj = -temp;
      }
      
      i += di;
      j += dj;
    }
    
    return result;
  }
  
   getArrowDirection(di, dj) {
    if (di === 0 && dj === 1) {
      return "→";
    } else if (di === 1 && dj === 0) {
      return "↓";
    } else if (di === 0 && dj === -1) {
      return "←";
    } else if (di === -1 && dj === 0) {
      return "↑";
    }
  }
  
   meanSquaredError(y, t) {
    let n = y.length;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += (y[i] - t[i]) ** 2;
    }
    return sum / n;
  }
     geneticLayerConstructor(numInputs, numOutputs) {
      // Define the weight and bias arrays
      const weights = [];
      for (let i = 0; i < numOutputs; i++) {
        const row = [];
        for (let j = 0; j < numInputs; j++) {
          row.push(Math.random());
        }
        weights.push(row);
      }
    
      const biases = [];
      for (let i = 0; i < numOutputs; i++) {
        biases.push(Math.random());
      }
    
      // Construct the layer object
      const layer = {
        weights: weights,
        biases: biases
      };
    
      return layer;
    }
    



     geneticModelCombiner(models) {
      // Choose a random model to use as the base
      const baseModel = models[Math.floor(Math.random() * models.length)];
      const combinedModel = {};
    
      // Iterate through the properties of the base model and combine them with the other models
      for (const property in baseModel) {
        // Skip over inherited properties
        if (!baseModel.hasOwnProperty(property)) continue;
    
        const values = models.map(model => model[property]);
    
        // If the property is an object, recursively combine its properties
        if (typeof baseModel[property] === 'object') {
          combinedModel[property] = this.geneticModelCombiner(values);
        } else {
          // Otherwise, choose a random value from the set of values for the property
          combinedModel[property] = values[Math.floor(Math.random() * values.length)];
        }
      }
    
      return combinedModel;
    }
    

    
 
    
    createModel(pretrained, ndim) {
      



    }


    async saveModel(model) {
      // Define a name for the model.

      //get full dir to model
      let dir = __dirname;
      const modelName = '.\\my_model';
  
      //write model to file sync
//write save handler to file




await fs.writeFileSync ( modelName + '.json', JSON.stringify(model.toJSON()) );

    }
   
    async loadModel(modelname) {
      // Define the name of the model to load.
      const modelName = modelname;
    
      // Load the model using the model name.

      //read model from file sync

      fs.readFileSync(modelName + '.json', 'utf8', (err, jsonString) => {
        if (err) {
            console.log("File read failed:", err)
            return false;
        }
       // console.log('File data:', jsonString)
       return jsonString;
    })
    return null;
    }



    //Backpropagation Through Time (BPTT):  10 sephirot
   bptt(x, y, a, h, c, Wf, Wi, Wc, Wo, Wy) {

    // Forward pass
    //write to this forwardArray;
    let tempforward = this.forward(x, Wf, Wi, Wc, Wo, Wy);
     this.temp[2] = tempforward[0];
      this.temp[3] = tempforward[1];
      this.temp[4] = tempforward[2];

    //[a, h, c]
    // Backward pass
    const [dWf, dWi, dWc, dWo, dWy] = this.backward(x, y, a, h, c, Wf, Wi, Wc, Wo, Wy);
  
    return [dWf, dWi, dWc, dWo, dWy];
   }

    forward(x, Wf, Wi, Wc, Wo, Wy) {
    // Forward pass
    const a = [];
    const h = [];
    const c = [];

    // Initialize the hidden state and cell state
    h[-1] = tf.zeros([this.hiddenSize, 1]);
    c[-1] = tf.zeros([this.hiddenSize, 1]);

    // Iterate through the time steps
    for (let t = 0; t < x.length; t++) {
      // Calculate the input gate, forget gate, output gate, and cell state
      const f = tf.sigmoid(tf.add(tf.matMul(Wf, x[t]), tf.matMul(Wf, h[t - 1])));
      const i = tf.sigmoid(tf.add(tf.matMul(Wi, x[t]), tf.matMul(Wi, h[t - 1])));
      const o = tf.sigmoid(tf.add(tf.matMul(Wo, x[t]), tf.matMul(Wo, h[t - 1])));
      const c_ = tf.tanh(tf.add(tf.matMul(Wc, x[t]), tf.matMul(Wc, h[t - 1])));

      // Calculate the cell state and hidden state
      c[t] = tf.add(tf.mul(f, c[t - 1]), tf.mul(i, c_));
      h[t] = tf.mul(o, tf.tanh(c[t]));

      // Calculate the output
      a[t] = tf.matMul(Wy, h[t]);
    }

    return [a, h, c];

  }

    backward(x, y, a, h, c, Wf, Wi, Wc, Wo, Wy) {

    // Backward pass

    // Initialize the gradients
    const dWf = tf.zerosLike(Wf);
    const dWi = tf.zerosLike(Wi);
    const dWc = tf.zerosLike(Wc);
    const dWo = tf.zerosLike(Wo);
    const dWy = tf.zerosLike(Wy);

    // Initialize the deltas
    let df = tf.zeros([this.hiddenSize, 1]);
    let di = tf.zeros([this.hiddenSize, 1]);
    let dc = tf.zeros([this.hiddenSize, 1]);
    let do_ = tf.zeros([this.hiddenSize, 1]);

    // Iterate through the time steps in reverse order
    for (let t = x.length - 1; t >= 0; t--) {

      // Calculate the output gradient
      const da = tf.sub(a[t], y[t]);

      // Calculate the hidden state gradient
      const dh = tf.add(tf.matMul(tf.transpose(Wy), da), dh_next);

      // Calculate the cell state gradient
      const dc = tf.add(tf.mul(dh, tf.tanh(c[t])), dc_next);

      // Calculate the forget gate gradient
      const df = tf.mul(dc, c[t - 1]);

      // Calculate the input gate gradient
      const di = tf.mul(dc, c_);

      // Calculate the output gate gradient
      do_ = tf.mul(dh, tf.tanh(c[t]));

      // Calculate the input gradient
      let dWx = tf.add(tf.matMul(df, tf.transpose(x[t])), tf.matMul(di, tf.transpose(x[t])), tf.matMul(dc_, tf.transpose(x[t])), tf.matMul(do_, tf.transpose(x[t])));

      // Calculate the hidden state gradient
      const dh_next = tf.add(tf.matMul(tf.transpose(Wf), df), tf.matMul(tf.transpose(Wi), di), tf.matMul(tf.transpose(Wc), dc_), tf.matMul(tf.transpose(Wo), do_));

      // Calculate the cell state gradient
      const dc_next = tf.mul(dc, f);

      // Update the gradients
      dWf = tf.add(dWf, tf.matMul(df, tf.transpose(h[t - 1])));
      dWi = tf.add(dWi, tf.matMul(di, tf.transpose(h[t - 1])));
      dWc = tf.add(dWc, tf.matMul(dc_, tf.transpose(h[t - 1])));
      dWo = tf.add(dWo, tf.matMul(do_, tf.transpose(h[t - 1])));
      dWy = tf.add(dWy, tf.matMul(da, tf.transpose(h[t])));
    }

    return [dWf, dWi, dWc, dWo, dWy];

  }


  train (pretrainedData, epochs, batchSize, learningRate, callback) {

    const optimizer = tf.train.adam(learningRate);

    for (let i = 0; i < epochs; i++) {
      const batch = this.getBatch(pretrainedData, batchSize);
      const loss = optimizer.minimize(() => {
        return this.loss(batch);
      }, true);

      if (callback) {
        callback(i, loss); //maybe new params, or rewrite?
      }
    }

  }

  getBatch (pretrainedData, batchSize) {
  //be like ngram
    const batch = [];
    for (let i = 0; i < batchSize; i++) {
      const index = Math.floor(Math.random() * pretrainedData.length); //random can fix on time or on balance or on health of ai pc
      batch.push(pretrainedData[index]);
    }

    return batch;
  }

  ngram (str, n) {

    try {
    if (str!==undefined) {
    const ngrams = [];
    for (let i = 0; i < str.length - n + 1; i++) {
      ngrams.push(str.substring(i, i + n));
    }

    return ngrams;
  }
  else {
    return [];
  }

} catch {}
  }

  loss (batch) {
    let loss = tf.scalar(0);
    for (let i = 0; i < batch.length; i++) {
      const [x, y] = this.preprocess(batch[i]);
      const [a, h, c] = this.forward(x);
      loss = tf.add(loss, this.calculateLoss(a, y));
    }

    return loss.div(tf.scalar(batch.length));
  }

  calculateLoss (a, y) {
    return tf.mean(tf.square(tf.sub(a, y)));
  }

  preprocess (str) {

    if (str===undefined) {
      return [];
    }
    const x = [];
    const y = [];
    const ngrams = this.ngram(str, this.n);
    for (let i = 0; i < ngrams.length - 1; i++) {
      const input = this.normalizeString(ngrams[i]);
      const output = this.normalizeString(ngrams[i + 1]);
      x.push(input);
      y.push(output);
    }

    return [x, y];
  }


step (index, loss = 1) {
    console.log(`Epoch: ${index}, Loss: ${loss}`);
this.lastBrainwave= this.harmonicMeanInRange(this.temp, 0, index) //maybe set  input size
 //maybe set  input size or balance value
this.short=this.bptt(this.temp[0], this.temp[1],  this.temp[2], this.temp[3], this.temp[4],
  this.temp[5], this. temp[6], this.temp[7], this.temp[8], this.temp[9]);
this.temp[5]=this.short[0];
this.temp[6]=this.short[1];
this.temp[7]=this.short[2];
this.temp[8]=this.short[3];
this.temp[9]=this.short[4];



  //need update temp
//this.temp
}
  // normalizeString (str) {
  //   const normalizedArr = [];
  //   for (let i = 0; i < str.length; i++) {
  //     const char = str[i];
  //     const codePoint = char.charCodeAt(0);
  //     const normalizedValue = codePoint / 0x10FFFF;
  //     normalizedArr.push(normalizedValue);
  //   }


    //get train from words pairs with asscoiativity - 3 float values
    trainAssociativity(word1, word2, asscoiativity) {

//neural network with 3 inputs and 1 output
//input - 3 float values
//output - 1 float value
//train with 3 float values and 1 float value
//train with 3 float values and 1 float value
//train with 3 float values and 1 float value
//train with 3 float values and 1 float value


    }

 





  }
  
  function normalizeString(str) {

    if(str) {
    const normalizedArr = [];
    //console.log(str);
    for (let i = 0; i < str.length; i++) {
      const char = str[i];
      const codePoint = char.charCodeAt(0);
      const normalizedValue = codePoint / 0x10FFFF;
      normalizedArr.push(normalizedValue);
    }
    return normalizedArr;
  } else {
    return [];
  }

  }
  function harmonicMeanInRange(arr, start=0, end=1) {
    let sum = 0;
    for (let i = start; i < end; i++) {
      sum += arr[i];
    }
    return sum / (end - start);
  }
  function harmonicalString(str) {
    let normalizedArr = normalizeString(str);

    return harmonicMeanInRange(normalizedArr);
  }

  function convert3floatToFlatTypedArray(float1, float2, float3) {
    let arr = [ input= [[float1], [float2]], output = [float3] ];
    let flat = arr.flat();
    let typed = new Float32Array(flat);
    return typed;
  }

  function convert3floatToNestedArrays(float1, float2, float3) {
    return { input: [[float1], [float2]], output: [float3] };
  }

  function prepareData(predata) {
    //подготовить данные 
    //split object to array by \t
  //convert 1 and 2 column to input, 3 to output
  // const data = [
  //   { input: [[0.1], [0.2]], output: [0.3] },
  //   { input: [[0.2], [0.3]], output: [0.5] },
  //   { input: [[0.3], [0.4]], output: [0.7] }
  // ];
  
  //split string to array by \t
    let data = predata.split("\n");
    //split array to array by \n
    data = data.map((item) => {
        let input = item.split("\t");
        let word1=harmonicalString(input[0]);
        let word2=harmonicalString(input[1]);
        let output=parseFloat(input[2]);
        //make to flat/TypedArray
let flat = convert3floatToFlatTypedArray(word1, word2, output);

return new Float32Array(flat);
       // return [ input = [[word1], [word2]], output = [output] ];
       // return { input: [[word1], [word2]], output: [output] };
        
        //return  { input: [[], [harmonicalString(input[1])]], output: [parseFloat(input[2])] };
    });

    
    console.log("prepared data\n");
    console.log(data);

  

//   data = data.map((item) => {
//     let input = item.split(" ");
//     let output = input.pop();
//     return { input: input, output: output };
//   });
  
  //console.log(data);
  
  
    //tensor2d

    let tensor2d = tf.tensor2d(data);
  
    
  return tensor2d;
  }
  



function writeToFile(data, filename) {
    fs.writeFile(filename, data, (err) => {

        // In case of a error throw err.
        if (err) throw err;
    })
}

function readFromFile(filename) {
//print current full dir
    console.log(__dirname);
    // Read the file and send to the callback
    fs.readFile(filename, function (err, data) {
        if (err) {
            return console.log(err);
        }
        return data;
    });
    return false;
}

// read from file synchronously
// function readFromFile(filename) {

//     Read the file and send to the callback

//     try {








function checkFileExists(filename) {

    // Check if the file "newfile.txt" exists in the current directory.

    if (fs.existsSync(filename)) {
        return true;
    }
    else {
        return false;
    }
}


function print(text) {

    console.log(text);
}

var mind = new MindPattern();
function pretrain(...args) {
    //get first arg
    let data = args[0];
    console.log(typeof data);

    
//   
//   //получить неподготовленные данные
let prepared=prepareData(data);
console.log(prepared.length);

//show ndim layer
console.log(prepared.shape);

let ndim = prepared.shape;

  
   let geneticModel = mind.geneticModelConstructor(16);
   console.log(geneticModel); 
   // Output: ['T', 'C', 'A', 'G', 'T', 'A', 'T', 'C', 'A', 'G']
  
// Define some example models to combine
const model1 = {
  inputSize: 100,
  hiddenLayers: {
    size: 50,
    activation: 'sigmoid'
  },
  outputSize: 1,
  learningRate: 0.01
};

const model2 = {
  inputSize: 50,
  hiddenLayers: {
    size: 25,
    activation: 'relu'
  },
  outputSize: 2,
  learningRate: 0.001
};

const model3 = {
  inputSize: 75,
  hiddenLayers: {
    size: 100,
    activation: 'tanh'
  },
  outputSize: 1,
  learningRate: 0.001
};

// Combine the models using the geneticModelCombiner function
let combinedModel = mind.geneticModelCombiner([model1, model2, model3]);

console.log(combinedModel);

   let geneticLayer = mind.geneticLayerConstructor(4, 3);
    console.log(geneticLayer); 
    






    // Output: { weights: [ [0.18, 0.23, 0.94, 0.49], [0.09, 0.79, 0.62, 0.34], [0.71, 0.06, 0.58, 0.47] ], biases: [0.84, 0.67, 0.25] }




// mind.train(prepared, 10, 3, 0.23, mind.step).this(function (err, result) {
//         if (err) {
//             console.log(err);
//         }

//         console.log(result);

// console.log(mind.lastBrainwave);
// console.log(mind.temp);
// comsole.log(mind.short);

//       });






//if (mind.model==null) {
 //mind.createModel(prepared, ndim);
//}
  

//console.log(global);


//get stack size
//console.log(process.resourceUsage());


//console.log(global.performance);




//console.log( v8.getHeapCodeStatistics() );

//console.log( v8.getHeapStatistics() );


// const maxCallStackSize = process.maxCallStackSize;
// console.log(maxCallStackSize);
// Performance {
//   nodeTiming: PerformanceNodeTiming {
//     name: 'node',
//     entryType: 'node',
//     startTime: 0,
//     duration: 2663.9416999816895,
//     nodeStart: 6.879700005054474,
//     v8Start: 15.617399990558624,
//     bootstrapComplete: 68.29899999499321,
//     environment: 35.73469999432564,
//     loopStart: -1,
//     loopExit: -1,
//     idleTime: 0
//   },
//   timeOrigin: 1678236012298.964
// }
// what from this is callstack size? or maxsize
//console.log(global.performance.nodeTiming.loopStart);
//print all keras functions and fields
//console.log(keras);

 //with reflection print code of adam optimizer
 //console.log(keras.optimizers);



}


print("start");

//check all files in current dir
fs.readdirSync('./').forEach(file => {
    console.log(file);
    })


print("check file exists");
let fileExists = checkFileExists("sociation.tsv");
print(fileExists);
var data = fs.readFileSync('sociation.tsv', 'utf8');
let pretrained = false;
//when data was read start pretrain
//if undefined  repeat 
pretrain(data);


        

app.get('/', (req, res) => res.send('Hello World!'));


app.post("/saveMemories", (req, res) => {
    console.log("saveMemories");
    res.send("saveMemories");
});

app.post("/loadMemories", (req, res) => {
    console.log("loadMemories");
  //read tsv file

   fs.readFile('sociation.tsv', 'utf8', function(err, data) {
    if (err) throw err;
    console.log(data);

    //convert to json
    var lines = data.split("\n");
    var result = [];
    var headers = lines[0].split("\t");

    for(var i=1;i<lines.length;i++){

        var obj = {};
        var currentline = lines[i].split("\t");

        for(var j=0;j<headers.length;j++){
            obj[headers[j]] = currentline[j];
        }

        result.push(obj);

    }

res.send(result);
    //res.send(data);

   });


});




app.listen(port, () => console.log(`Example app listening on port ${port}!`));
