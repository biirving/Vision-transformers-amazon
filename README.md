# Vision-transformers-amazon
Library for the Vision Transformers Benchmarking project. 

Within this repository, there are example files displaying benchmarking and compilation for various batch sizes on Inf.2x instances, as well as a g5 instance for comparison.

For efficient use of the accuracy measurements within this repository, ensure that you have the ImageNet dataset downloaded to an accessible folder, preferrably within this cloned repository. Create a ticket or ask a SA on the team to help with access. 

The repositories should be run inside of an Amazon Inf.2x instance, set up with a pytorch environment. See this link for the first part of the setup process. 
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-intro/pytorch-setup/pytorch-install.html

For the Inferentia compilations and benchmarking, setup a virtual environment or a container, with all of the requirements in requirements_inferentia.txt downloaded. 
pip -r requirements_inferentia.txt

The GPU compilations and benchmarking require a different pytorch version that is compatible with the G5 ec2 instances. 
It is reccommended to create a new virtual environment, and install the requirements for gpu.
pip -r requirements_gpu.txt

This command will install a compatible pytorch version. This command has to be run separately.
pip install torch --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu116

In the Inferentia folder, there is an example jupyter notebook which runs through the scripts for BEiT. The process can be emulated by running the files individually within the environment. There are no example notebooks for LayoutLMv3 or ViT, but there are files available that have to be run separately.

To run the files separately to compile, measure, and benchmark on Inferentia:

For ViT:

1.) Run vitCompile.py inside the compile Vision-transformers-amazon/Inferentia/compile folder
python vitCompile.py

2.) Run vitBenchmark.py inside the Vision-transformers/Inferentia/benchmark folder
python vitBenchmark.py

3.) Run the inferentiaAccuracy.py file with the correct model path inserted. (If no path variables are changed, the model file should lie in the compile folder where the compilation was initially ran)

python inferentiaAccuracy.py 


