### Datasets

 - *Netflix* is no longer officially distributed, but can be found via bit-torrent
 - *R2 - Yahoo! Music User Ratings of Songs with Artist, Album, and Genre Meta Information, v. 1.0 (1.4 Gbyte & 1.1 Gbyte)* is available from [Yahoo Webscope](http://webscope.sandbox.yahoo.com/catalog.php?datatype=r)

### Setup

 - a cluster with an [Apache Hadoop](http://hadoop.apache.org) installation, we used version 1.0.4
 - as we leverage [JBlas](http://mikiobraun.github.io/jblas/) for solving the linear equations involved in ALS, your cluster machines need to have the [necessary fortran libraries](https://github.com/mikiobraun/jblas/wiki/Missing-Libraries)  installed
 - the datasets need to be converted to text files where each line has the following format: *userID[TAB]itemID[TAB]rating* and copied to HDFS
