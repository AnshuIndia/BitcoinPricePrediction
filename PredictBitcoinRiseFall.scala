/* Load the prediction model and use it to predict the
rise/fall in Bitcoin's price the next day using number of
*tweets and number of google searches two day ago. The model
* is trained this way. */

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.sql.SparkSession


object PredictRiseFall {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("PredictRiseFall");

    val sc = new SparkContext(conf);

    val spark = SparkSession
      .builder()
      .appName("PredictRiseFall")
      .config("spark.some.config.option", "some-value")
      .getOrCreate();

    val predictionModel = LinearRegressionModel.load(sc, "/Users/Anshu/Documents/scalaLinearRegressionWithSGDModelNew4");

    /* Predict Rise/Fall in Bitcoin's price */
    /* Use today's predicted price and tomorrow's predicted
    price to predict rise/fall in Bitcoin's price
    Day1.csv and Day2.csv needs to be updated daily with
    new values of number of google searches and number of tweets
    related to Bitcoin. This files contain data for two seperate
    da*/
    val GTDataRDD1 = sc.textFile("/Users/Anshu/Downloads/Day1.csv");
    val GTDataRDD2 = sc.textFile("/Users/Anshu/Downloads/Day2.csv");


    val DayRDD1 = GTDataRDD1.map {line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble , Vectors.dense(parts(1).toDouble, parts(2).toDouble))
    }

    val DayRDD2 = GTDataRDD2.map {line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble , Vectors.dense(parts(1).toDouble, parts(2).toDouble))

    }

    val predictedPriceToday = DayRDD1.map { point =>
      val prediction = predictionModel.predict(point.features)
      (prediction)
    }

    val  predictedPriceTomorrow = DayRDD2.map { point =>
      val prediction = predictionModel.predict(point.features)
      (prediction)
    }

    val result = predictedPriceTomorrow.first() - predictedPriceToday.first();
    println("Result" + result);

    if(result< 0){
      val PRes = "Bitcoin Price will FALL tomorrow by "  + result.toString;
      scala.tools.nsc.io.File("/Users/Anshu/Documents/ResultFile").writeAll(PRes);

    }
    else{
      val PRes = "Bitcoin Price will RISE tommorow by " + result.toString ;
      scala.tools.nsc.io.File("/Users/Anshu/Documents/ResultFile").writeAll(PRes);
    }
  }
}