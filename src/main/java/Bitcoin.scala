import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.streaming.Seconds
import org.apache.spark.streaming.StreamingContext
import scala.util.parsing.json.JSON
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.log4j.Logger
import org.apache.log4j.Level
import scalaj.http.Http
import scalaj.http.HttpOptions

case class Record(timestamp:Double,source:String,volume:Double,price:Double)

object Bitcoin {


	def main(args:Array[String]){
	  
	  Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
		val conf = new SparkConf().setMaster("local[60]").setAppName("bitcoin") 
				val sc = new SparkContext(conf)


		val ssc = new StreamingContext(sc,Seconds(60))
		val lines = ssc.socketTextStream("api.bitcoincharts.com", 27007)

		val jsonValue = lines.map(JSON.parseFull(_))
		.map(_.get.asInstanceOf[scala.collection.immutable.Map[String, Any]])

		val fields = jsonValue.map(m=>Record(m("timestamp").toString.toDouble,m("symbol").toString,m("volume").toString.toDouble,m("price").toString.toDouble)) 
		val data = fields.filter { x => x.source.contains("USD") } 
		data.print() 
		
//		var timestamp: Double = 0;
//		var volume : Double = 0;
//		var price : Double  = 0;
//		
//		var counter : Int= 0
//		
//		val sum = data.map{
//		  x=> timestamp = x.timestamp
//		      volume = volume + x.volume.toDouble
//		      price = price + x.price.toDouble 
//		      counter = counter + 1
//		}
//		
//		
//		var averageVolume : Double = volume/counter
//		var averagePrice : Double = price/counter
//		println("timestamp : "+timestamp)
//		println("volume : "+averageVolume)
//		println("price : "+averagePrice)
//		volume = 0;
//		price = 0;
//		timestamp = 0;
//		val labelPoint = new LabeledPoint(timestamp,)
		

		val finalData = data.map{ 
			x => LabeledPoint(0 , Vectors.dense(x.timestamp,x.volume,x.volume,x.volume,x.price))
		}
		
		val model = LinearRegressionModel.load(sc, "linearRegression")
		
				val pred = data.map{
			x => val prediction = model.predict(Vectors.dense(x.timestamp,x.volume,(x.volume*x.price),x.price)) 
			println("Pred: "+ prediction/(60*1000)) 
			(x.price,prediction) 
		}

		pred.foreachRDD{
			x=> x.collect()
					println(x)
//					val result = Http("http://example.com/url").postData("""{"data":"+x+"}""")
//                      .header("Content-Type", "application/json")
//                      .header("Charset", "UTF-8")
//                      .option(HttpOptions.readTimeout(10000)).asString
		} 
		ssc.start() 
		ssc.awaitTermination()
	}


}