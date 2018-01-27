import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.tuning.TrainValidationSplitModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD


object JoinData {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("JoinData");

    val sc = new SparkContext(conf);

    val spark = SparkSession
      .builder()
      .appName("JoinData")
      .config("spark.some.config.option", "some-value")
      .getOrCreate();

    import spark.implicits._

    /* Number of transactions per day*/
   /* val txnDF = spark.read.csv("/Users/Anshu/Documents/formattedTransactionData.csv"); */

    /* Daily Bitcoin Price */
    val bitcoinDF = spark.read.option("header","true").csv("/Users/Anshu/Documents/formattedBitcoinDataNew.csv");

    /* Daily number of tweets with "bitcoin" as a keyword */
    val tweetDF = spark.read.option("header","true").csv("/Users/Anshu/Downloads/tweetF.csv");

    tweetDF.show(2);

    /* Number of daily google searches about bitcoin */
    val googleTrendsDF = spark.read.option("header","true").csv("/Users/Anshu/Downloads/Bitcoin_GoogleTrendsC.csv");


    /* Name the columns */
   /* val transactionDF = txnDF.selectExpr("_c0 as Date","_c1 as NumberOfTransactions","_c2 as ChangeInNumberOfTransaction"); */

   /* val transactionDFNew = transactionDF.selectExpr("Date","cast(NumberOfTransactions as Double) NumberOfTransactions"); */


   /* val bitcoinDF = btcDF.selectExpr("_c0 as Date","_c1 as BitcoinPrice","_c2 as ChangeInBitcoinPrice"); */

   /* val bitcoinDFNew = bitcoinDF.selectExpr("Date","cast(BitcoinPrice as Double) BitcoinPrice","cast(ChangeInBitcoinPrice as Double) ChangeInBitcoinPrice"); */

    val bitcoinDFNew = bitcoinDF.selectExpr("Date","cast(BitcoinPrice as Double) BitcoinPrice");

    bitcoinDFNew.show(2);

    /*val bitcoinDF = btcDF.selectExpr("_c0 as Date","_c1 as BitcoinPrice"); */

    val tweetFeedDF = tweetDF.selectExpr("Date","cast(Tweets as Double) Tweets");

    //val googleTrend = googleTrendsDF.selectExpr("_c0 as Date","_c1 as BitcoinSearches");

    val googleTrends = googleTrendsDF.selectExpr("Date","cast(BitcoinSearches as Double) BitcoinSearches");

    googleTrends.show(2);


     /* txnMax = transactionDFNew.select("NumberOfTransactions").rdd.map(r =>r.getAs[Double]("NumberOfTransactions")).max(); */
     /*val txnMax = transactionDFNew.select("NumberOfTransactions").rdd.max(); */
    /* val priceMax = bitcoinDFNew.select("BitcoinPrice").rdd.map(r =>r.getAs[Double]("BitcoinPrice")).max();
     val googleSearchMax = googleTrends.select("GoogleTrends").rdd.map(r=>r.getAs[Double]("GoogleTrends")).max(); */
    /*val txnMax = transactionDFNew.select(max("NumberOfTransactions")).getAs[Int]; */
   /*println("PRINT VALUES");
    println(txnMax);
    println("   ");
    println(priceMax);
    println("   ");
    println(googleSearchMax); */



    /*Join transaction and price data */
    /*val joinTransactionAndPrice = transactionDF.join(bitcoinDFNew,Seq("Date")); */

    //Join transaction-price data with tweet data
    val joinPriceAndTweet  = bitcoinDFNew.join(tweetFeedDF,Seq("Date"));

    /* Join transaction-price-tweet data with Google trends */
    val joinPriceTweetAndTrends = joinPriceAndTweet.join(googleTrends,Seq("Date"));

    joinPriceTweetAndTrends.show(2);
    joinPriceTweetAndTrends.printSchema();

    val correlation = joinPriceTweetAndTrends.stat.corr("BitcoinPrice","BitcoinSearches");

    val corr2 = joinPriceTweetAndTrends.stat.corr("BitcoinPrice","Tweets");

    println("CORRELATION Between Price and searches" + correlation);
    println("CORRELATION Between Price and tweets" + corr2);

    val finalDat = joinPriceTweetAndTrends.selectExpr("cast(BitcoinPrice as Double) BitcoinPrice", "cast(Tweets as Double) Tweets", "cast(BitcoinSearches as Double) GoogleSearch")

    finalDat.write.format("csv").mode("overwrite").save("/Users/Anshu/Documents/joinedData.csv");

    val rawDataRDD = sc.textFile("/Users/Anshu/Documents/joinedData.csv");


   /* val tempDataRDD = spark.read.option("header","true").csv("/Users/Anshu/Downloads/finalNormal.csv");

    val finalData = tempDataRDD.selectExpr("cast(Bitcoin as Double) BitcoinPrice", "cast(TwitterN as Double) Tweets", "cast(GoogleN as Double) GoogleSearch");

    finalData.write.format("csv").mode("overwrite").save("/Users/Anshu/Documents/finalNormal.csv");

    val rawDataRDD1 = sc.textFile("/Users/Anshu/Documents/finalNormal.csv");

    finalData.printSchema(); */

    val NumOfDataPoints = rawDataRDD.count();
    println("Data points are "+ NumOfDataPoints);


    /*Split the RDD into two RDDs (Testing and Training Data with 80 percent being
    *training data and the remaining data is for testing the model)*/
    val splitTrainingAndTesting = rawDataRDD.randomSplit(Array(0.8,0.2),2);

     /*Parse the training RDD*/
    /* Label = BitcoinPrice
       Feature Added = Number of bitcoin searches on google
       Feature to be added = Number of tweets with keyword "bitcoin"
     */
    val parsedTrainingRDD = rawDataRDD.map {line =>
                                          val parts = line.split(',')
                                          LabeledPoint(parts(0).toDouble , Vectors.dense(parts(1).toDouble , parts(2).toDouble ))
    }.cache() ;

    /*Parse the testing data */
    /*
    val parsedTestingRDD = splitTrainingAndTesting(1).map {line =>
      val parts = line.split(',')
      LabeledPoint(parts(1).toDouble , Vectors.dense(parts(2).toDouble, parts(3).toDouble))
    }.cache() ; */

    val cnt = parsedTrainingRDD.count();
    println("Count is " + cnt);

    parsedTrainingRDD.foreach(println);



    /* Paramaters for model generation */
    val numIterations = 100
    val stepSize = 0.0001
     /*val model = LinearRegressionWithSGD.train(parsedTrainingRDD, numIterations, stepSize); */

    val algorithm = new LinearRegressionWithSGD()
    algorithm.setIntercept(true)
    algorithm.optimizer
      .setNumIterations(numIterations)
      .setStepSize(stepSize)

    /* DONE : Model trained for number of bitcoin  searches on google
    * TO DO : Train the model on number of tweets and number of tweets
    * FEATURE REMOVED AFTER ANALYSIS : The model does not fit on number of bitcoin transactions.*/

    val model = algorithm.run(parsedTrainingRDD);

    /* Coefficient and intercept of the model */
    println("weights: %s, intercept: %s".format(model.weights, model.intercept));

    println("MODEL IS::  ")
    println(model);

    /*val lrModel = lr.fit(parsedTrainingRDD); */

    /* Evaluate model on training examples and compute training error */


    val valuesAndPreds = parsedTrainingRDD.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    /*Save the predicted values */
    valuesAndPreds.saveAsTextFile("/Users/Anshu/Documents/PredictedValue");

    valuesAndPreds.foreach(println);

    val MSE = valuesAndPreds.map{ case(v, p) => math.pow((p - v), 2) }.mean()
    println("training Mean Squared Error = " + MSE)
    /*val predictionAndLabel = valuesAndPreds.zip(parsedTestingRDD.map(._label)) */

    // Save and load model
    model.save(sc, "/Users/Anshu/Documents/scalaLinearRegressionWithSGDModelNew1");
   val sameModel = LinearRegressionModel.load(sc, "/Users/Anshu/Documents/scalaLinearRegressionWithSGDModelNew1");

    println("Print here")
/* To predict the value of bitcoin when number of tweets are 105.01K and google searches are 600K. */
println(sameModel.predict(Vectors.dense(105.01, 600)));


}
}
