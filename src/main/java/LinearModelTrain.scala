
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.feature.StandardScaler

object LinearModelTrain {


	def main(args: Array[String]) = {

		val conf = new SparkConf()
		.setAppName("BitcoinLinearRegression")
		.setMaster("local")
		val sc = new SparkContext(conf)

		// Load and parse the data
		val data = sc.textFile("/home/acer/Downloads/DataSource.csv")

		val parsedData = data.map { line =>
		val parts = line.split(',')
		LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).toDouble,
				parts(2).toDouble, 
				parts(3).toDouble,
				parts(4).toDouble))
		}.cache()

		val scaler = new StandardScaler(withMean = true, withStd = true)
		.fit(parsedData.map(x => x.features))
		val scaledData = parsedData
		.map(x => 
		LabeledPoint(x.label, 
				scaler.transform(Vectors.dense(x.features.toArray))))

				val numIterations = 1000000
				val step = 0.07899
				val algorithm = new LinearRegressionWithSGD()
    		algorithm.setIntercept(true)
    		algorithm.optimizer.setNumIterations(numIterations).setStepSize(step).setMiniBatchFraction(0.25)

		val model = algorithm.run(scaledData)
		//model.save(sc, "linearRegression")
		val valuesAndPreds = scaledData.map { point =>
		val prediction = model.predict(point.features)
		println("Label: "+point.label)
		println("prediction:"+prediction)
		(point.label, prediction)
		}

		// To save the result in a text file
		//valuesAndPreds.saveAsTextFile("/home/acer/Downloads/results1.txt")

		val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
				println("Calculated Mean Squared Error = " + MSE)

	}
}