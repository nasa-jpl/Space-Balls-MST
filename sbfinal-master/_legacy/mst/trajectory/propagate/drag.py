import Monte as M
import mpy.units as units
from ...config.earth.atmosphere import set_atmos

AtmModel, SolarFluxMag = set_atmos()


# forces: drag

def load_drag(boa, sc, forces):

    # forces: drag

    # check params are set for sc: shape, area, mass, drag coef:
    # done in sc setup above

    #   drag model
    fluxdata = open(AtmModel).readlines()
    fluxUnit = M.UnitDbl(1e-22,'kg/sec/sec')
    fluxTable = M.MonthlyTableFlux( boa, 'Flux Model')
    yearInitialized = False
    eps = 1e-9
    for i, line in enumerate(fluxdata):
        if i >= 7:
            data = line.strip().split()
            #print( "data: 0: ",data[0], " data[1] ", data[1] )
            if data[1] == 'JAN':
                fluxYear = int(float(data[0]))
                fluxVals = [ M.HermiteTable( [0.05, 0.5, 0.95], [float(data[4]), float(data[3]), float(data[2])]).interp( SolarFluxMag )[0]*fluxUnit ]
                magVals = [ M.HermiteTable( [0.05, 0.5, 0.95], [float(data[7]), float(data[6]), float(data[5])]).interp( SolarFluxMag )[0] ]
                yearInitialized = True
            elif yearInitialized:
                fluxVals.append( M.HermiteTable( [0.05, 0.5, 0.95], [float(data[4]), float(data[3]), float(data[2])]).interp( SolarFluxMag )[0]*fluxUnit )
                magVals.append( M.HermiteTable( [0.05, 0.5, 0.95], [float(data[7]), float(data[6]), float(data[5])]).interp( SolarFluxMag )[0] )
        
            if data[1] == 'DEC' and yearInitialized:
                fluxTable.insertFlux( fluxYear, fluxVals )
                fluxTable.insertGeo( fluxYear, magVals )

    forces.append(M.AtmDragForce)

    densityModel = M.Dtm(boa,'DTM','Flux Model' )
    atm = M.AtmDrag(boa, sc.name, sc.cd,  sc.name + ' Shape', 'DTM') # 'OCO2 Shape', 'DTM')

    print ('The mass is %.1f kg.' %( sc.mass.value() ) )
    print ('The area is %.1f m^2.' %( sc.area.value()*1.e6 ))
    print ('The solar flux is at %.0f%%.' %( SolarFluxMag*100. ))
    print ('drag loaded')

    return atm
    