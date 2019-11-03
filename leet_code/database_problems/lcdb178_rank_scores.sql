/*
 Leetcode Database Practice
 Difficulty: Medium
 Url: https://leetcode.com/problems/rank-scores/

 Write a SQL query to rank scores.
 If there is a tie between two scores, both should have the same ranking.
 Note that after a tie, the next ranking number should be the next consecutive integer value.
 In other words, there should be no "holes" between ranks.
 */
SELECT
	*,
	DENSE_RANK() OVER (ORDER BY score DESC) rnk
FROM
	leetcode.scores;