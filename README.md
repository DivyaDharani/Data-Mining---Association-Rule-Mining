# Association Rule Mining
* Determining anomalous events through Association Rule Mining

This is one of the Data Mining tasks performed on the sensor data of an Artificial Pancreas System. Refer to the code repo https://github.com/DivyaDharani/Data-Mining---Machine-Learning-Model/ for details about meal and no-meal data as well as about the features extracted. 

The aim of this project is to determine the anomalous events in the meal data using Association Rule Mining.

* Extracted slope and frequency domain features from the meal data (refer to the project link given above)

* Discretized the Continuous Glucose Monitor’s values of the dataset into multiple bins and considered bin number of the max CGM value (B<sub>max</sub>), bin number of the CGM value at the start of the meal (B<sub>meal</sub>), and insulin bolus value for the meal time (I<sub>B</sub>) for each meal data row as item sets

* Used **Apriori** algorithm to extract rules of the form {B<sub>max</sub>, B<sub>meal</sub>} → I<sub>B</sub> and found the anomalous rules by picking the rules having confidence below a particular threshold
