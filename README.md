# Wild-ID-
A real-time deep learning based acoustic classification of different wildlife species. My objective was to build a deep learning model capable of identifying different animal species based on the sounds they make. My model is more specialised in identifying the different bird species because I trained it on a dataset containing 10127 species of birds all over the planet. The significant challenge in a model like this are immense diversity of species, environmental background noise, and the varying volume of wildlife species. Coming to know about these challenges led me to search for extensive features like PCEN, etc., from some well-established research papers. One of the facts that I found quite interesting is that the model visualises sound, i.e., it makes a 2D picture called a spectrogram out of the audio file and learns the behaviour of different audio patterns in it to identify the animals. The objective was also to train my model in such a way that it can understand variations in sounds and adjust itself to various recording qualities.

My project consists of Model A and Model B. For Model A, which is a general classifying system of basic animals and environmental sounds,  I have used a basic environment dataset for it. Based on that, my model A will classify what sound it is and predict its label. Furthermore, if the label corresponds to a bird, the information will trigger Model B, which is a pure specialist in identifying Birds which even a human can't normally differentiate between. 

For proper information relating to architecture and my model, please refer to my end-term report 

You can find my detailed architecture in the files that I have uploaded (both the Python files and also the end-term report, which I have written)

To use the code for my Model A, refer to the code in the folder esc 50 initial training code. I personally ran 50 epochs on it to train my model 

To use the code for my Model B, refer to the 2 folders initial_model and new_model, which contain 2 model architectures on which I had trained. I have also attached the .pth files of learnt weights from my training process in all the folders for you to refer to.
