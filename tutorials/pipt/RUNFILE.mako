<%!
import numpy as np
%>

-- *------------------------------------------*
-- *                                          *
-- * base grid model with input parameters    *
-- *                                          *
-- *------------------------------------------*
RUNSPEC

TITLE
 INVERTED 5 SPOT MODEL

--DIMENS
-- NDIVIX NDIVIY NDIVIZ
--   60    60      5 /

--BLACKOIL
OIL
WATER
GAS
DISGAS

METRIC

TABDIMS
-- NTSFUN  NTPVT  NSSFUN  NPPVT  NTFIP  NRPVT  NTENDP
     1       1      35      30     5     30      1 /

EQLDIMS
-- NTEQUL  NDRXVD  NDPRVD
   1       5       100 /

WELLDIMS
-- NWMAXZ NCWMAX NGMAXZ MWGMAX 
    15     15     2      20 /

VFPPDIMS
-- MXMFLO MXMTHP MXMWFR MXMGFR MXMALQ NMMVFT
   10     10     10     10     1      1 /

VFPIDIMS
-- MXSFLO MXSTHP NMSVFT
   10     10     1 /

AQUDIMS
-- MXNAQN  MXNAQC NIFTBL NRIFTB NANAQU NCAMAX
   0       0      1      36     2       200/

FAULTDIM
    500 / 

START
 01 JAN 2022 /

NSTACK
 25 /


NOECHO

GRID
INIT

INCLUDE
 '../Grid.grdecl' /
/

PERMX
% for i in range(0, len(permx)):
% if permx[i] < 6:
${"%.3f" %(np.exp(permx[i]))}
% else:
${"%.3f" %(np.exp(6))}
% endif
% endfor
/

COPY
 'PERMX'  'PERMY'  /
 'PERMX'  'PERMZ' /
/


PROPS    ===============================================================

INCLUDE
 '../pvt.txt' /
/

REGIONS  ===============================================================

ENDBOX

SOLUTION ===============================================================


--    DATUM  DATUM   OWC    OWC    GOC    GOC    RSVD   RVVD   SOLN
--    DEPTH  PRESS  DEPTH   PCOW  DEPTH   PCOG  TABLE  TABLE   METH
EQUIL
     2355.00 200.46 3000 0.00  2355.0 0.000     /

 
RPTSOL
'PRES' 'SWAT' /

RPTRST
 BASIC=2 /



SUMMARY ================================================================

RUNSUM


RPTONLY

WWIR
 'INJ1'
 'INJ2'
 'INJ3'
/

WOPR
 'PRO1'
 'PRO2'
 'PRO3'
/

WWPR
 'PRO1'
 'PRO2'
 'PRO3'
/

ELAPSED

SCHEDULE =============================================================


RPTSCHED
 'NEWTON=2' /

RPTRST
 BASIC=2 FIP RPORV /

------------------- WELL SPECIFICATION DATA --------------------------

INCLUDE
'../Schdl.sch' /
/


WCONINJE
'INJ1' WATER 'OPEN' BHP 2* 300/
'INJ2' WATER 'OPEN' BHP 2* 300/
'INJ3' WATER 'OPEN' BHP 2* 300/
/

WCONPROD
 'PRO1' 'OPEN' BHP 5* 90 /
 'PRO2' 'OPEN' BHP 5* 90 /
 'PRO3' 'OPEN' BHP 5* 90 /
/
--------------------- PRODUCTION SCHEDULE ----------------------------



TSTEP
10*400 /
/

END
