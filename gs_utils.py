import os
from google.cloud import storage


GOOGLE_APPLICATION_CREDENTIALS = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')


def get_gs_client():
    return storage.Client()


def get_bucket(bucket):
    client = get_gs_client()
    return client.get_bucket(bucket)


def upload_file_to_gcs(path_to_file, bucket_name, save_path):
    """
    stores a blob in gcs
    """
    gs_bucket = get_bucket(bucket_name)
    gs_bucket.blob(save_path).upload_from_filename(path_to_file)
    
    return


def upload_bytes_to_gcs(data_stream, bucket_name, save_path):
    
    gs_bucket = get_bucket(bucket_name)
    gs_bucket.blob(save_path).upload_from_string(data_stream)

    
def download_blob(bucket_name, source_blob_name, destination_file_name):
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    
    return blob