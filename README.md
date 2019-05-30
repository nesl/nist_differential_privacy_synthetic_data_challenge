## Differentially Private Dataset Release using Wasserstein GANs

This repo contains an implementation for the award-winning solution to the [2018 Differential Privacy Synthetic Data Challenge](https://www.nist.gov/communications-technology-laboratory/pscr/funding-opportunities/prizes-challenges/2018-differential) by team UCLANESL.

Our solution has been awarded the 5th place in [Match#3](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=17421&pm=15315) of the challenge and an earlier version has also won the 4th place in [Match #1](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=17319&pm=15124).
Here is the [press release](https://www.nist.gov/communications-technology-laboratory/pscr/funding-opportunities/prizes-challenges/2018-differential) for winners announcement.

The solution trains a wasserstein generative adversarial network (w-GAN) that is trained on the real private dataset.
Differentially private training is applied by santizing (norm clipping and adding Gaussian noise) the gradients of the discriminator.
Once the model is trained, it can be used to generate sytnethic dataset by feeding random noise into the generator.


## Team Members
* [*Prof. Mani Srivastava*](http://nesl.ee.ucla.edu/people/1) [(msrivastava)](https://github.com/msrivastava)  - Team Captain (Match 1 and Match 3)
* [*Moustafa Alzantot*](http://web.cs.ucla.edu/~malzantot/) [(malzantot)](https://github.com/malzantot)  - (Match 1 and Match 3)
* [*Nat Snyder* - (natsnyder1)](https://github.com/natsnyder1) - Match 1
* [*Supriyo Charkaborty* (supriyogit)](https://github.com/supriyogit) - Match 1


----------------------------

This project is maintained by [(malzantot)](https://github.com/malzantot)  (malzantot@ucla.edu)


---
## Technical Report

Please find our technical report at [`reports/UCLANESL_solution_privacy_proof.pdf`](reports/UCLANESL_solution_privacy_proof.pdf).

If you would like to cite our work in your own research, please use the following:
```
@misc{uclanesl_dp_wgan,
    author       = {Alzantot, Moustafa and Srivastava, Mani},
    title        = {{Differential Privacy Synthetic Data Generation using WGANs}},
    year         = 2019,
    version      = {1.0},
    url          = {https://github.com/nesl/nist_differential_privacy_synthetic_data_challenge/}
    }
```


## Instructions
Our solution uses the TensorFlow library which is GPU-accelerated deep learning framework.
The runtime is signficantlly improved when running on a GPU- powered machine.
It has been tested to work successfully on a machine with Titan X GPU. 

#### Steps to run without Docker

You can run the code by first installing the requirments:

`
pip3 install -r requirements.txt
`
then, run:

```
bash run_uclanesl.sh colorado.csv output.csv colorado-specs.json epsilon [delta]
```

where epsilon is the required epsilon and delta is an optional parameter for the delta value to generate

(espilon, delta) differentially private dataset. If delta parameter is not provided then the default value will be

(1/n^2) where n is the number of rows in input data.


### Steps to run using `Nvidia-docker`:


first, build the docker image:

```
docker build -t uclanesl_wgan .
```
then, run the image while giving the appropriate argument values.

```
docker run --runtime=nvidia -v $(pwd):/x uclanesl_wgan /x/colorado.csv /x/output.csv /x/colorado-specs.json  epsilon [delta]
```

