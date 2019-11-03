/*
 Leetcode Database Practice
 Difficulty: Easy
 Url: https://leetcode.com/problems/combine-two-tables/

 Write a SQL query for a report that provides the following information for each person in the Person table,
 regardless if there is an address for each of those people:
 (FirstName, LastName, City, State)
*/
SELECT
	p.FirstName,
    p.LastName,
    a.City,
    a.State
FROM
	person p
	LEFT JOIN address a
	ON p.personid=a.personid