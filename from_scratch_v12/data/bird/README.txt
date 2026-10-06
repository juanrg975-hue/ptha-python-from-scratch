Plate convergence tables read by step 3 (lib/bird_convergence.py)
===================================================================

generate.py --convergence picks one of the two (default: bird).

PB2002_steps.dat.txt.zip          --convergence bird
------------------------
Bird, P. (2003) An updated digital model of plate boundaries.
Geochemistry, Geophysics, Geosystems 4(3), doi:10.1029/2001GC000252
Public supplementary data of that paper (Peter Bird's website:
http://peterbird.name/publications/2003_PB2002/2003_PB2002.htm), unchanged.
One line per boundary step: the two end points, the relative velocity and
its divergent (Div_vel) and right-lateral (RL_vel) components in mm/yr, and
Bird's boundary type. Step 3 uses the steps of the convergent types SUB,
OCB and CCB.

bird_griffin_traces_table.csv.zip   --convergence bird-griffin
---------------------------------
The table PTHA18 itself used (Davies & Griffin 2018), copied unchanged from
rptha so that no step has to read the rptha folder:
    rptha/R/examples/austptha_template/DATA/BIRD_PLATE_BOUNDARIES/
        sourcezone_traces_table_merged.csv.zip
(rptha: https://github.com/GeoscienceAustralia/ptha, BSD-3 licence.)
It merges three sources (column 'collator'):
    Bird2003_subset  1060 of Bird's steps that PTHA18 kept
    JG               2645 source-zone traces with plate rates put together
                     by Jonathan Griffin for eastern Indonesia and other
                     zones (reformat_edited_Bird_convergence.R), plus the
                     outer-rise traces (class 'Normal')
    GD               100 rows added by Gareth Davies (Arakan 23 mm/yr,
                     Seram south 1 mm/yr)
On zones where PTHA18 used Bird's own steps both tables give the same
convergence; on zones where it used Griffin's traces they can differ a lot
(newguinea2: 38 mm/yr with Bird, 96 with Bird+Griffin). Which zone used
which: html/docs/convergence_sources.html.
