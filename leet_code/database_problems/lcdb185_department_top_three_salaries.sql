/*
 Leetcode Database Practice
 Difficulty: Hard
 Url: https://leetcode.com/problems/department-top-three-salaries/

 The Employee table holds all employees. Every employee has an Id, and there is also a column for the department Id.
 The Department table holds all departments of the company.
 Write a SQL query to find employees who earn the top three salaries in each of the department.
 For the above tables, your SQL query should return the following rows (order of rows does not matter).
 */
with merged as (
    select
        d.name as Department,
        e.name as Employee,
        e.salary as Salary
    from employees e
    left join department d
    on e.departmentid=d.id
),
ranked as (
    select
        Department,
        Employee,
        Salary,
        dense_rank() OVER (partition by department order by salary desc) as rnk
    from merged
)
select * from ranked where rnk <= 3;