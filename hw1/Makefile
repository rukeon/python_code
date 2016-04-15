#####
# Student ID: 2010-11304
# Description: makefile for Homework 1. # indicates a comment in a bash shell
# Usage: 
#   * make in: run IndegreCounter.jar on Hadoop to compute the in-degree for each node
#   * make out: run OutdegreeCounter.jar on Hadoop to compute the out-degree for each node
#   * make in_dist: run DegreeDistribution.jar on Hadoop to compute the in-degree distribution
#   * make out_dist: run DegreeDistribution.jar on Hadoop to compute the out-degree distribution 

all: in out in_dist out_dist in_journal in_dist_journal out_journal out_dist_journal

in:
	rm -rf indegreeoutput
	rm -rf indegreeoutput.txt	
	# compile
	hadoop com.sun.tools.javac.Main IndegreeCounter.java
	# jar
	jar cf IndegreeCounter.jar IndegreeCounter*.class
	# run
	hadoop jar IndegreeCounter.jar IndegreeCounter ./problem.edge ./indegreeoutput
	hadoop fs -cat indegreeoutput/part-r-00000
	hadoop fs -get indegreeoutput/part-r-00000 indegreeoutput.txt 

out:
	rm -rf outdegreeoutput
	rm -rf outdegreeoutput.txt
	# compile
	hadoop com.sun.tools.javac.Main OutdegreeCounter.java
	# jar
	jar cf OutdegreeCounter.jar OutdegreeCounter*.class
	# run
	hadoop jar OutdegreeCounter.jar OutdegreeCounter ./problem.edge ./outdegreeoutput
	hadoop fs -cat outdegreeoutput/part-r-00000
	hadoop fs -get outdegreeoutput/part-r-00000 outdegreeoutput.txt

in_dist:
	rm -rf indegreedistribution
	rm -rf indegreedistribution.txt
	# compile
	hadoop com.sun.tools.javac.Main DegreeDistribution.java
	# jar
	jar cf DegreeDistribution.jar DegreeDistribution*.class
	# run
	hadoop jar DegreeDistribution.jar DegreeDistribution ./indegreeoutput.txt ./indegreedistribution
	hadoop fs -cat indegreedistribution/part-r-00000
	hadoop fs -get indegreedistribution/part-r-00000 indegreedistribution.txt

out_dist:
	rm -rf outdegreedistribution
	rm -rf outdegreedistribution.txt
	# compile
	hadoop com.sun.tools.javac.Main DegreeDistribution.java
	# jar
	jar cf DegreeDistribution.jar DegreeDistribution*.class
	# run
	hadoop jar DegreeDistribution.jar DegreeDistribution ./outdegreeoutput.txt ./outdegreedistribution
	hadoop fs -cat outdegreedistribution/part-r-00000
	hadoop fs -get outdegreedistribution/part-r-00000 outdegreedistribution.txt

in_journal:
	rm -rf in-degree-journal
	# compile
	hadoop com.sun.tools.javac.Main IndegreeCounter.java
	# jar
	jar cf IndegreeCounter.jar IndegreeCounter*.class
	# run
	hadoop jar IndegreeCounter.jar IndegreeCounter ./soc-LiveJournal1.txt ./in-degree-journal
	hadoop fs -cat in-degree-journal/part-r-00000
	hadoop fs -get in-degree-journal/part-r-00000 indegree-journal-output.txt

in_dist_journal:
	rm -rf indegreedistribution-journal
	rm -rf indegreedistribution-journal.txt
	# compile
	hadoop com.sun.tools.javac.Main DegreeDistribution.java
	# jar
	jar cf DegreeDistribution.jar DegreeDistribution*.class
	# run
	hadoop jar DegreeDistribution.jar DegreeDistribution ./indegree-journal-output.txt ./indegreedistribution-journal
	hadoop fs -cat indegreedistribution-journal/part-r-00000
	hadoop fs -get indegreedistribution-journal/part-r-00000 indegreedistribution-journal.txt

out_journal:
	rm -rf out-degree-journal
	# compile
	hadoop com.sun.tools.javac.Main OutdegreeCounter.java
	# jar
	jar cf OutdegreeCounter.jar OutdegreeCounter*.class
	# run
	hadoop jar OutdegreeCounter.jar OutdegreeCounter ./soc-LiveJournal1.txt ./out-degree-journal
	hadoop fs -cat out-degree-journal/part-r-00000
	hadoop fs -get out-degree-journal/part-r-00000 outdegree-journal-output.txt

out_dist_journal:
	rm -rf outdegreedistribution-journal
	rm -rf outdegreedistribution-journal.txt
	# compile
	hadoop com.sun.tools.javac.Main DegreeDistribution.java
	# jar
	jar cf DegreeDistribution.jar DegreeDistribution*.class
	# run
	hadoop jar DegreeDistribution.jar DegreeDistribution ./outdegree-journal-output.txt ./outdegreedistribution-journal
	hadoop fs -cat outdegreedistribution-journal/part-r-00000
	hadoop fs -get outdegreedistribution-journal/part-r-00000 outdegreedistribution-journal.txt

