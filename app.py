from flask import Flask, render_template, request, url_for
from pyspark.sql import SparkSession
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, Word2VecModel
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
import pyspark.sql.functions as F
import numpy as np

app = Flask(__name__)

spark = SparkSession.builder.getOrCreate()

# Load Word2Vec model, data with vectors, topic modeling and summarization data
model_path = "/Users/celine/Desktop/5430nlp/group/web/word_embedding_model"
model = Word2VecModel.load(model_path)
stored_vectors_df = spark.read.format("parquet").load("/Users/celine/Desktop/5430nlp/group/web/dataset_with_vectors.parquet")
topic_sum_photo = spark.read.format("parquet").load("/Users/celine/Desktop/5430nlp/group/web/topic_sum_photo.parquet")

# Cache the data
stored_vectors_df.cache()
topic_sum_photo.cache()

# Function to calculate cosine similarity
def cossim(v1, v2):
    if np.dot(v1, v1) == 0 or np.dot(v2, v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / np.sqrt(np.dot(v1, v1)) / np.sqrt(np.dot(v2, v2)))

# Register UDF
cosine_similarity = udf(cossim, FloatType())

@app.route('/', methods=['GET', 'POST'])
def index():
    search_results = []
    search_results_with_photos = []
    show_results = False 

    if request.method == 'POST':
        show_results = True 
        # Capture search bar input
        search_query = request.form['search_bar']
        rating_filter = request.form.get('rating-select')

        # Tokenization and stop word removal
        regextok = RegexTokenizer(gaps=False, pattern='\w+', inputCol='inputText', outputCol='tokens')
        stopwrmv = StopWordsRemover(inputCol='tokens', outputCol='tokens_sw_removed')

        # Process the input
        query_df = spark.createDataFrame([(1, search_query)]).toDF('index', 'inputText')
        query_tok = regextok.transform(query_df)
        query_swr = stopwrmv.transform(query_tok)

        # Transform the query using Word2Vec model
        query_vec = model.transform(query_swr)
        query_vec = query_vec.select('wordvectors').first()['wordvectors']

        # Add the query vector as a column to the DataFrame
        stored_vectors_df_with_query = stored_vectors_df.withColumn("queryVec", F.array([F.lit(float(v)) for v in query_vec]))

        # Compute similarity
        stored_vectors_df_with_similarity = stored_vectors_df_with_query.withColumn("similarity", cosine_similarity(F.col("wordvectors"), F.col("queryVec")))
        # Compute average similarity for each restaurant and select top 20 in Philadelphia
        avg_similarities_df = stored_vectors_df_with_similarity.groupBy("business_id", "name") \
            .agg(F.mean("similarity").alias("avg_similarity")) \
            .orderBy(F.desc("avg_similarity")) \
            .limit(20)
        top_results = avg_similarities_df.collect()


        # filter the dataset for these top business IDs
        top_business_ids = [row['business_id'] for row in top_results]
        filtered_df = topic_sum_photo.filter(topic_sum_photo.business_id.isin(top_business_ids))
        # Join with the top results 
        result_df = filtered_df.join(spark.createDataFrame(top_results), "business_id", "right")

        # Add the full URL to each result's dictionary
        for row in result_df.collect():
            result_dict = row.asDict()
            # Generate the URL for the static path
            result_dict['photo_url'] = url_for('static', filename=f'photos/{result_dict["photo_id"]}.jpg')
            search_results_with_photos.append(result_dict)

        # Filter results if rating filter is applied
        if rating_filter:
            rating_filter = float(rating_filter)
            search_results_with_photos = [result for result in search_results_with_photos if result['stars'] >= rating_filter]

        # Sort the results by the rating in descending order
        search_results_with_photos = sorted(search_results_with_photos, key=lambda x: x['stars'], reverse=True)


    # Render the template with the modified results
    return render_template('index.html', search_results=search_results_with_photos, show_results=show_results)
    

if __name__ == '__main__':
    app.run(debug=True, port=5002)
