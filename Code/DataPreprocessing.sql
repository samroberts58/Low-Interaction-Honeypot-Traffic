/* Data Preprocessing & Data Set Refinement*/

SELECT * FROM FINAL --10,933,525 rows
----------------------------------------------------------------------------------------------------------------------------
SELECT src_port, count(src_port) as 'Total' from FINAL group by src_port order by 1  -- 60,652 rows. 7,300,300 are blank and 2,063 are '0'. Drop column.

SELECT dst_port, count(dst_port) as 'Total' from FINAL group by dst_port order by 2   -- 786 rows. 4,953,858 are blank - need to drop rows

SELECT ts, count(ts) as 'Total' from FINAL group by ts order by 1  -- 22,017 rows.  

SELECT src_ip, count(src_ip) as 'Total IP Addresses' from Final group by src_ip order by 1 --137,181 rows. 2 are " - need to drop rows

SELECT session, count(session) as 'Total Sessions' from FINAL group by session order by 1 -- 879,557 rows

SELECT duration, count(duration) as 'Total durations' from FINAL group by duration order by 1 -- 790,650 rows. 10,028,542 are blank, 2 are '0'.  Drop column.

SELECT eventid, count(eventid) as 'Total Events' from FINAL group by eventid order by 1 -- 17 rows. 212,598 rows are blank - need to drop rows

SELECT sensor, count(sensor) as 'Total sensors' from FINAL group by sensor order by 1 -- 59 rows

SELECT Country, count(Country) as 'Total Country' from Final group by Country order by 1 -- 188 rows. 561 are '#N/A' - need to drop, 37 are '0' - need to drop rows

SELECT City, count(City) as 'Total City' from Final group by City order by 1  -- 10,792 rows. 559 are '#N/A', 2 are '#N/A,#N/A', 855,312 are '0'. Update all to 'N/A'


----------------------------------------------------------------------------------------------------------------------------
/* Drop rows as needed */
-- src_port column deleted
-- duration column deleted

-- dst_port
DELETE FROM FINAL WHERE dst_port = ''

-- eventid
DELETE FROM FINAL WHERE eventid = ''

-- Country
DELETE FROM FINAL WHERE Country = '#N/A'
DELETE FROM FINAL WHERE Country = '0'


-- City
UPDATE Final
SET City = 'N/A'
WHERE City in ('#N/A', '#N/A,#N/A', '0')

----------------------------------------------------------------------------------------------------------------------------
/* Additional data quality checks */

SELECT * FROM FINAL WHERE ts not like '%/%/%'   -- Pass

SELECT * FROM FINAL WHERE src_ip not like '%.%' -- Pass

SELECT * FROM FINAL WHERE len(session) < 4 -- Pass, most seem to have 8, as number gets smaller, suspect leading zeroes were dropped by CSV - Just need to be unique & consistent

SELECT * FROM FINAL WHERE (session like '#N/A') or (session like '0') -- Pass
---------------------------------------------------------------------------------------------------------------------------
SELECT * FROM FINAL  -- 5,895,631 rows

-- Remove the Netherlands
DELETE FROM FINAL WHERE Country = 'Netherlands'

-- Final Result
SELECT * FROM FINAL -- 852,981 rows
