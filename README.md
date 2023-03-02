# Readme
## _The Last Markdown Editor, Ever_

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Dillinger is a cloud-enabled, mobile-ready, offline-storage compatible,
AngularJS-powered HTML5 Markdown editor.

- Type some Markdown on the left
- See HTML in the right
- ✨Magic ✨

## Features

- Import a HTML file and watch it magically convert to Markdown
- Drag and drop images (requires your Dropbox account be linked)
- Import and save files from GitHub, Dropbox, Google Drive and One Drive
- Drag and drop markdown and HTML files into Dillinger
- Export documents as Markdown, HTML and PDF

Markdown is a lightweight markup language based on the formatting conventions
that people naturally use in email.
As [John Gruber] writes on the [Markdown site][df1]

> The overriding design goal for Markdown's
> formatting syntax is to make it as readable
> as possible. The idea is that a
> Markdown-formatted document should be
> publishable as-is, as plain text, without
> looking like it's been marked up with tags
> or formatting instructions.

This text you see here is *actually- written in Markdown! To get a feel
for Markdown's syntax, type some text into the left window and
watch the results in the right.

## Tech

Dillinger uses a number of open source projects to work properly:

- [AngularJS] - HTML enhanced for web apps!
- [Ace Editor] - awesome web-based text editor
- [markdown-it] - Markdown parser done right. Fast and easy to extend.
- [Twitter Bootstrap] - great UI boilerplate for modern web apps
- [node.js] - evented I/O for the backend
- [Express] - fast node.js network app framework [@tjholowaychuk]
- [Gulp] - the streaming build system
- [Breakdance](https://breakdance.github.io/breakdance/) - HTML
to Markdown converter
- [jQuery] - duh

And of course Dillinger itself is open source with a [public repository][dill]
 on GitHub.

## Installation

Dillinger requires [Node.js](https://nodejs.org/) v10+ to run.

Install the dependencies and devDependencies and start the server.

```sh
cd dillinger
npm i
node app
```

For production environments...

```sh
npm install --production
NODE_ENV=production node app
```

## Hyperparameters

Here are the introduced hyperparameters:
| Hyperparameters     | Usage                                                                                           |
|---------------------|-------------------------------------------------------------------------------------------------|
| device              | run the calculation on CPU(-1) or GPU(0) (default:0, else:-1)                                   |
| dataset             | dataset to apply (default:mnist, else:YinYang)                                                  |
| action              | train or test or visualization (unsupervised_ep, else:supervised_ep, visu)                      |
| epochs              | number of epochs to train (default:30)                                                          |
| batchSize           | training batch size (default:128)                                                               |
| test_batchSize      | testing batch size (default:256)                                                                |
| dt                  | time discretization (default:0.2)                                                               |
| T                   | number of time steps in the free phase (default:40)                                             |
| Kmax                | number of time steps in the nudging phase (default:50)                                          |
| beta                | nudging parameter (default:0.5)                                                                 |
| clamped             | clamped states of the network to avoid divergence (default:1, else:0)                           |
| convNet             | whether use the convolutional layers                                                            |
| C_list              | channel list (default:[1,32,64])                                                                |
| padding             | padding or not (default: 0, else:1)                                                             |
| convF               | convolution filter size (default:5)                                                             |
| Fpool               | pooling filter size (default:2)                                                                 |
| fcLayers            | fully connected layer (default:[784, 512, 500])                                                 |
| lr                  | learning rate (default:[0.02, 0.006])                                                           |
| activation_function | activation function (default: hardsigm, else: tanh, sigmoid)                                    |
| Optimizer           | optimizer to update the gradients (default:SGD, else:Adam)                                      |
| errorEstimate       | two different way to estimate the gradients (default:one-sided, else:symmetric)                 |
| lossFunction        | type of loss function (default:MSE, else:Cross-entropy)                                         |
| eta                 | coefficient for regulating memory capacity in the on-line homeostasis (default:0.6)             |
| gamma               | coefficient for regulating the homeostasis effect (default:0.5)                                 |
| nudge_N             | number of winners to be nudged (default:1)                                                      |
| n_class             | number of class in dataset (default:10)                                                         |
| class_activation    | activation function of the added classification layer (default:softmax, else:hardsigm, sigmoid) |
| class_lr            | learning rate of the classification layer (default:0.045)                                       |
| class_epoch         | epoch for training the classification layer (default:30)                                        |
| class_Optimizer     | optimizer used for the classification layer (default:Adam, else:SGD)                            |
| coeffDecay          | coefficient of learning rate decay (default:1, else:0.7)                                        |
| epochDecay          | epoch to decay the learning rate (default:10, else:5, 15)                                       |
| gammaDecay          | coefficient of homeostasis decay (default:1, else:0.6)                                          |
| weightNormalization | whether to use the weight normalization (default:0, else:1)                                     |
| randomHidden        | whether update only the weights of output layers (default:0, else:1)                            |
| Dropout             | whether to use the dropout (default:0, else:1)                                                  |
| dropProb            | decide the probability of dropout (default:[0.1, 0.2])                                          |
| torchSeed           | generate the reproductive result (default:0, else: number of seed)                              |
| analysis_preTrain   | whether to load the trained model (default:0, else:1)                                           |
| imWeights           | whether we imshow the weights of synapses (default:0, else:1)                                   |
| maximum_activation  | whether to draw the maximum activation input for each neuron (default:0, else:1)                |
| imShape             | the size for each imshow of weights (default: [28, 28, 32, 32])                                 |
| display             | decide the number of neurons whose visualization are presented (default:[10, 10, 10, 10])       |

## Development

Want to contribute? Great!

Dillinger uses Gulp + Webpack for fast developing.
Make a change in your file and instantaneously see your updates!

Open your favorite Terminal and run these commands.

First Tab:

```sh
node app
```

Second Tab:

```sh
gulp watch
```

(optional) Third:

```sh
karma test
```

#### Building for source

For production release:

```sh
gulp build --prod
```

Generating pre-built zip archives for distribution:

```sh
gulp build dist --prod
```

## Docker

Dillinger is very easy to install and deploy in a Docker container.

By default, the Docker will expose port 8080, so change this within the
Dockerfile if necessary. When ready, simply use the Dockerfile to
build the image.

```sh
cd dillinger
docker build -t <youruser>/dillinger:${package.json.version} .
```

This will create the dillinger image and pull in the necessary dependencies.
Be sure to swap out `${package.json.version}` with the actual
version of Dillinger.

Once done, run the Docker image and map the port to whatever you wish on
your host. In this example, we simply map port 8000 of the host to
port 8080 of the Docker (or whatever port was exposed in the Dockerfile):

```sh
docker run -d -p 8000:8080 --restart=always --cap-add=SYS_ADMIN --name=dillinger <youruser>/dillinger:${package.json.version}
```

> Note: `--capt-add=SYS-ADMIN` is required for PDF rendering.

Verify the deployment by navigating to your server address in
your preferred browser.

```sh
127.0.0.1:8000
```

## License

MIT

**Free Software, Hell Yeah!**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
