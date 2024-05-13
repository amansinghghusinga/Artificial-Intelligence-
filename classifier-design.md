In this classifier i used my pre trained data that i got from my training model and text vectorization. First we clean the data and then transform that into numerical data using pre trained vectorizer model.

To get the predictions for genre I passed my cleaned data through neural networks and then compared those probabilities with the threshold which is 0.5 to give the final genre for that description.I also used Lime to give a proper explanation into which word is used for the classification.

I used this design as the neural netowrk is able to get all complex patterns in the text which is really benefical to undersdtand the descriptions. Tensorflow provide robust farmework for my classifier and jolib is used to handl text data.

