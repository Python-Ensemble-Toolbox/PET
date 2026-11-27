
-------------------------------------------------------------------------------
-- DATAFILE FOR ECLIPSE TESTING
-------------------------------------------------------------------------------


---------------------------- Runspec Section ----------------------------------
--NOECHO

RUNSPEC

TITLE
 50x50x1=2,500 Eclipse test example

DIMENS
   50   50    1  /

-- Phases present

OIL

WATER

GAS

DISGAS

-- Units

METRIC

-- Table dimension

TABDIMS

-- NoSatTabl     MaxNodesSatTab  MaxFIPReg      MaxSatEndpointsDepthTab
--       NoPVTTab       MaxPressNodes    MaxRsRvNodes
     1       1      20      200       1      200       1    /


-- Well dimension

WELLDIMS
-- MaxNo  MaxPerf  MaxGroup MaxWell/Group
    5     1        5       5  /


START
  1 'JAN' 2000 /


--FMTOUT

--UNIFIN
--UNIFOUT

--NOSIM

NSTACK
 10 /

------------------------------- Grid Section ----------------------------------

GRID

-- Including the indiviual grid file

INCLUDE
 '../include/50X50X1.COORD' /

INCLUDE
 '../include/50X50X1.ZCORN' /

--INCLUDE
-- 'TRUE_PORO' /

PORO
2500*0.2
/

INCLUDE
 '../include/PERMX' /

--PERMX
--2500*500
--/

COPY
 PERMX PERMY /
 PERMX PERMZ /
/

MULTIPLY
 PERMZ 0.001 /
/

--GRIDFILE
-- 2 1 /

--NOGGF

INIT

NEWTRAN


------------------------------- Edit Section ----------------------------------


------------------------------ Properties Section -----------------------------


PROPS

ROCK
-- RefPressure          Compressibility
-- for PoreVol Calc
--BARSA                 1/BARSA
  300                   1.450E-05 /

INCLUDE
 '../include/ALL.PVO' /  

INCLUDE
 '../include/ALL.RCP' /



------------------------------- Regions Section -------------------------------


------------------------------ Solution Section -------------------------------

SOLUTION

EQUIL
2000.000 200.00 2280.00  .000  2000.000  .000     1      0       0 /

PBVD
 1    10 
 1000 10 /

--RPTSOL
-- RESTART /

--RPTRST
-- BASIC=2 /


------------------------------- Summary Section -------------------------------

SUMMARY

------------------------------------------------
--Output of production data/pressure  for FIELD:
------------------------------------------------

FOPR
FWPR
FLPR
FLPT
FOPT
FGPT
FWPT
FPR
FWIT

-------------------------------------------------
-- Gas and oil in place:
-------------------------------------------------

FOIP
FGIP

-----------------------------------------
--Output of production data for all wells:
-----------------------------------------
WOPR
WWPR
WGPR
WWCT
WGOR
WTHP
/
WWIR
 'INJ1'
 'INJ2'
 'INJ3'
 'INJ4'
/
WBHP
 'INJ1'
 'INJ2'
 'INJ3'
 'INJ4'
/
WPI
 'INJ1'
 'INJ2'
 'INJ3'
 'INJ4'
/


FVIR
FVPR
RPTONLY

RUNSUM

SEPARATE

RPTSMRY
 1  /

DATE


TCPU

------------------------------ Schedule Section -------------------------------

SCHEDULE

SKIPREST

RPTRST
 BASIC=5 DEN/

WELSPECS
 INJ1 G1 2 2 2000 WATER /
 INJ2 G1 49 2 2000 WATER /
 INJ3 G1 2 49 2000 WATER /
 INJ4 G1 49 49 2000 WATER /
 PROD G1 25 25 2000 WATER /
/

COMPDAT
--Name I J K1 K2 STATUS 2* RW
 INJ1 2 2 1 1   OPEN   2* 0.25 /
 INJ2 49 2 1 1 OPEN 2* 0.25 /
 INJ3 2 49 1 1 OPEN 2* 0.25 /
 INJ4 49 49 1 1 OPEN 2* 0.25 /
 PROD 25 25 1 1 OPEN 2* 0.25 /
/

--WPIMULT
-- INJ1 0.025 1* 1* 1* 1* 1* /
-- INJ3 0.025 1* 1* 1* 1* 1*/
--/


WCONPROD
PROD   OPEN BHP 5* 150 /
/

<%
import pandas as pd
years  = pd.date_range('2000-01-01', '2008-01-01', freq='YS').to_pydatetime()
report = pd.date_range('2000-01-01', '2008-01-01', freq='MS').to_pydatetime()
index  = 0
%>

%for date in report[:-1]:

%if date in years:
${'WCONINJE'}
${f'INJ1 WATER OPEN RATE {rate_inj1[index]} 1* 500.0  /'}
${f'INJ2 WATER OPEN RATE {rate_inj2[index]} 1* 500.0  /'}
${f'INJ3 WATER OPEN RATE {rate_inj3[index]} 1* 500.0  /'}
${f'INJ4 WATER OPEN RATE {rate_inj4[index]} 1* 500.0  /'}
${'/'}
<% index = index + 1 %>
%endif

${'TSTEP'}
${f'{pd.Period(str(date)).days_in_month} /'}

%endfor

END


