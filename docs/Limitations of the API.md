# Limitations

- Image datasets are supported in OpenML as a workaround by using a CSV file with image paths. This is not ideal and might eventually be replaced by something else. At the moment, the focus is on tabular data.
- OpenML-Tensorflow API currently only supports runs on image datasets. Other modalities will be included in the future.   
- Many features (like custom metrics, models etc) are still dependant on the OpenML Python API, which is in the middle of a major rewrite. Until that is complete, this package will not be able to provide all the features it aims to.
