"""Descriptive description."""

__author__ = {'TM', 'TB', 'ML'}

# standardrp.py
import numpy as np
import sys
import multiprocessing as mp

from numpy.random import poisson

# internal load
from misc.system_tools.environ_var import OpenBlasSingleThread  # Single threaded OpenBLAS runs


class elasticproperties:
    """
    Calculate elastic properties from standard
    rock-physics models, specifically following Batzle
    and Wang, Geophysics, 1992, for fluid properties, and
    Report 1 in Abul Fahimuddin's thesis at Universty of
    Bergen (2010) for other properties.

    Example
    -------
    >>> porosity = 0.2
    ... pressure = 5
    ... phases = ["Oil","Water"]
    ... saturations = [0.3, 0.5]
    ...
    ... satrock = Elasticproperties()
    ... satrock.calc_props(phases, saturations, pressure, porosity)
    """

    def __init__(self, input_dict):
        self.dens = None
        self.bulkmod = None
        self.shearmod = None
        self.bulkvel = None
        self.shearvel = None
        self.bulkimp = None
        self.shearimp = None
        # The overburden for each grid cell must be
        # specified as values on an .npz-file whose
        # name is given in input_dict.
        self.input_dict = input_dict
        self._extInfoInputDict()

    def _extInfoInputDict(self):
        # The key word for the file name in the
        # dictionary must read "overburden"
        if 'overburden' in self.input_dict:
            obfile = self.input_dict['overburden']
            npzfile = np.load(obfile)
            # The values of overburden must have been
            # stored on file using:
            # np.savez(<file name>,
            # obvalues=<overburden values>)
            self.overburden = npzfile['obvalues']
            npzfile.close()
        #else:
        #    # Norne litho pressure equation in Bar
        #    P_litho = -49.6 + 0.2027 * Z + 6.127e-6 * Z ** 2  # Using e-6 for scientific notation
        #    # Convert reservoir pore pressure from Bar to MPa
        #    P_litho *= 0.1
        #    self.overburden = P_litho

        if 'baseline' in self.input_dict:
            self.baseline = self.input_dict['baseline']  # 4D baseline
        if 'parallel' in self.input_dict:
            self.parallel = self.input_dict['parallel']

    def _filter(self):
        bulkmod = self.bulkimp
        self.bulkimp = bulkmod.flatten()

    def setup_fwd_run(self, state):
        """
        Setup the input parameters to be used in the PEM simulator. Parameters can be an ensemble or a single array.
        State is set as an attribute of the simulator, and the correct value is determined in self.pem.calc_props()

        Parameters
        ----------
        state : dict
            Dictionary of input parameters or states.

        Changelog
        ---------
        - KF 11/12-2018
        """
        # self.inv_state = {}
        # list_pem_param =[el for el in [foo for foo in self.pem['garn'].keys()] + [foo for foo in self.filter.keys()] +
        #                 [foo for foo in self.__dict__.keys()]]

        # list_tot_param = state.keys()
        # for param in list_tot_param:
        #    if param in list_pem_param or (param.split('_')[-1] in ['garn', 'rest']):
        #        self.inv_state[param] = state[param]

        pass

    def calc_props(self, phases, saturations, pressure,
                   porosity, dens = None, wait_for_proc=None, ntg=None, Rs=None, press_init=None, ensembleMember=None):
        ###

        #
        if not isinstance(phases, list):
            phases = [phases]
        if not isinstance(saturations, list):
            saturations = [saturations]
        if not isinstance(pressure, list) and \
                type(pressure).__module__ != 'numpy':
            pressure = [pressure]
        if not isinstance(porosity, list) and \
                type(porosity).__module__ != 'numpy':
            porosity = [porosity]
        #
        # Load "overburden" pressures into local variable to
        # comply with remaining code parts
        poverburden = self.overburden

        # debug
        self.pressure = pressure
        self.peff = poverburden - pressure
        self.porosity = porosity

        if press_init is None:
            p_init = self.p_init
        else:
            p_init = press_init

        # Average number of contacts that each grain has with surrounding grains
        coordnumber = self._coordination_number()

        # porosity value separating the porous media's mechanical and acoustic behaviour
        phicritical = self._critical_porosity()


        # Check that no. of phases is equal to no. of
        # entries in saturations list
        #
        assert (len(saturations) == len(phases))
        #
        # Make saturation a Numpy array (so that we
        # can easily access the values for each
        # phase at one grid cell)
        #
        # Transpose makes it a no. grid cells x phases
        # array
        saturations = np.array(saturations).T
        #
        # Check if we actually inputted saturation values
        # for a single grid cell. If yes, we redefine
        # saturations to get it on the correct form (no.
        # grid cells x phases array).
        #
        if saturations.ndim == 1:
            saturations = \
                np.array([[x] for x in saturations]).T
        #
        # Loop over all grid cells and calculate the
        # various saturated properties
        #
        self.phases = phases

        self.dens = np.zeros(len(saturations[:, 0]))
        self.bulkmod = np.zeros(len(saturations[:, 0]))
        self.shearmod = np.zeros(len(saturations[:, 0]))
        self.bulkvel = np.zeros(len(saturations[:, 0]))
        self.shearvel = np.zeros(len(saturations[:, 0]))
        self.bulkimp = np.zeros(len(saturations[:, 0]))
        self.shearimp = np.zeros(len(saturations[:, 0]))

        if ntg is None:
            ntg = [None for _ in range(len(saturations[:, 0]))]
        if Rs is None:
            Rs = [None for _ in range(len(saturations[:, 0]))]
        if p_init is None:
            p_init = [None for _ in range(len(saturations[:, 0]))]


        if dens is not None:
            assert (len(dens) == len(phases))
            # Transpose makes it a no. grid cells x phases array
            dens = np.array(dens).T

        #
        denss, bulks, shears = self._solidprops_Johansen()

        for i in range(len(saturations[:, 0])):
            #
            # Calculate fluid properties
            #
            if dens is None:
                densf, bulkf = \
                    self._fluidprops_Wood(self.phases,
                                          saturations[i, :], pressure[i], Rs[i])
            else:
                densf = self._fluid_dens(saturations[i, :], dens[i, :])

                bulkf = self._fluidprops_Brie(self.phases, saturations[i, :], pressure[i])
            #
            #denss, bulks, shears = \
            #    self._solidprops(porosity[i], ntg[i], i)

            #
            # Calculate dry rock moduli
            #

            #bulkd, sheard = \
            #    self._dryrockmoduli(porosity[i],
            #                        overburden[i],
            #                        pressure[i], bulks,
            #                        shears, i, ntg[i], p_init[i], denss, Rs[i], self.phases)
            #
            peff = self._effective_pressure(poverburden[i], pressure[i])


            bulkd, sheard = \
                self._dryrockmoduli_Smeaheia(coordnumber, phicritical, porosity[i], peff, bulks, shears)

            # -------------------------------
            # Calculate saturated properties
            # -------------------------------
            #
            # Density (kg/m3)
            #
            self.dens[i] = (porosity[i]*densf +
                            (1-porosity[i])*denss)
            #
            # Moduli (MPa)
            #
            self.bulkmod[i] = \
                bulkd + (1 - bulkd/bulks)**2 / \
                (porosity[i]/bulkf +
                 (1-porosity[i])/bulks -
                 bulkd/(bulks**2))
            self.shearmod[i] = sheard
            #
            # Velocities (km/s)
            #
            self.bulkvel[i] = \
                np.sqrt((abs(self.bulkmod[i]) +
                             4*self.shearmod[i]/3)/(self.dens[i]))
            self.shearvel[i] = \
                np.sqrt(self.shearmod[i] /
                            (self.dens[i]))
            #
            # convert from (km/s) to (m/s)
            #
            self.bulkvel[i] *= 1000
            self.shearvel[i] *= 1000
            #
            # Impedance (m/s)*(kg/m3)
            #
            self.bulkimp[i] = self.dens[i] * \
                self.bulkvel[i]
            self.shearimp[i] = self.dens[i] * \
                self.shearvel[i]



    def getMatchProp(self, petElProp):
        if petElProp.lower() == 'density':
            self.match_prop = self.getDens()
        elif petElProp.lower() == 'bulk_modulus':
            self.match_prop = self.getBulkMod()
        elif petElProp.lower() == 'shear_modulus':
            self.match_prop = self.getShearMod()
        elif petElProp.lower() == 'bulk_velocity':
            self.match_prop = self.getBulkVel()
        elif petElProp.lower() == 'shear_velocity':
            self.match_prop = self.getShearVel()
        elif petElProp.lower() == "bulk_impedance":
            self.match_prop = self.getBulkImp()
        elif petElProp.lower() == 'shear_impedance':
            self.match_prop = self.getShearImp()
        else:
            print("\nError in getMatchProp method")
            print("No model output type selected for "
                  "data match.")
            print("Legal model output types are "
                  "(case insensitive):")
            print("Density, bulk modulus, shear "
                  "modulus, bulk velocity,")
            print("shear velocity, bulk impedance, "
                  "shear impedance")
            sys.exit(1)
        return self.match_prop

    def getDens(self):
        return self.dens

    def getBulkMod(self):
        return self.bulkmod

    def getShearMod(self):
        return self.shearmod

    def getBulkVel(self):
        return self.bulkvel

    def getShearVel(self):
        return self.shearvel

    def getBulkImp(self):
        return self.bulkimp

    def getShearImp(self):
        return self.shearimp

    def getOverburdenP(self):
        return self.overburden

    def getPressure(self):
        return self.pressure

    def getPeff(self):
        return self.peff

    def getPorosity(self):
        return self.porosity
    #
    # ===================================================
    # Fluid properties start
    # ===================================================
    #
    def _fluidprops_Wood(self, fphases, fsats, fpress, Rs=None):
        #
        # Calculate fluid density and bulk modulus
        #
        #
        # Input
        #       fphases - fluid phases present; Oil
        #                 and/or Water and/or Gas
        #       fsats   - fluid saturation values for
        #                 fluid phases in "fphases"
        #       fpress  - fluid pressure value (MPa)
        #       Rs      - Gas oil ratio. Default value None

        #
        # Output
        #       fdens - density of fluid mixture for
        #               pressure value "fpress" inherited
        #               from phaseprops)
        #       fbulk - bulk modulus of fluid mixture for
        #               pressure value "fpress" (unit
        #               inherited from phaseprops)
        #
        # -----------------------------------------------
        #
        fdens = 0.0
        fbinv = 0.0

        for i in range(len(fphases)):
            #
            # Calculate mixture properties by summing
            # over individual phase properties
            #
            pdens, pbulk = self._phaseprops(fphases[i],
                                            fpress, Rs)
            fdens = fdens + fsats[i]*abs(pdens)
            fbinv = fbinv + fsats[i]/abs(pbulk)
        fbulk = 1.0/fbinv
        #
        return fdens, fbulk
    #
    # ---------------------------------------------------
    #

    def _fluid_dens(self, fsatsp, fdensp):
        fdens = sum(fsatsp * fdensp)
        return fdens

    def _fluidprops_Brie(self, fphases, fsats, fpress, Rs=None, e = 5):
        #
        # Calculate fluid density and bulk modulus BRIE et al. 1995
        # Assumes two phases liquid and gas
        #
        # Input
        #       fphases - fluid phases present; Oil
        #                 and/or Water and/or Gas
        #       fsats   - fluid saturation values for
        #                 fluid phases in "fphases"
        #       fpress  - fluid pressure value (MPa)
        #       Rs      - Gas oil ratio. Default value None
        #       e       - Brie's exponent (e= 5 Utsira sand filled with brine and CO2
        #                 Figure 7 in Carcione et al. 2006 "Physics and Seismic Modeling
        #                 for Monitoring CO 2 Storage"

        #
        # Output
        #       fbulk - bulk modulus of fluid mixture for
        #               pressure value "fpress" (unit
        #               inherited from phaseprops)
        #
        # -----------------------------------------------
        #


        for i in range(len(fphases)):
            #
            if fphases[i].lower() in ["oil", "wat"]:
                fsatsl = fsats[i]
                pbulkl = self._phaseprops_Smeaheia(fphases[i], fpress, Rs)
            elif fphases[i].lower() in ["gas"]:
                pbulkg = self._phaseprops_Smeaheia(fphases[i], fpress, Rs)


        fbulk = (pbulkl - pbulkg) * (fsatsl)**e + pbulkg

        #
        return fbulk
    #
    # ---------------------------------------------------
    #
    def _phaseprops_Smeaheia(self, fphase, press, Rs=None):
        #
        # Calculate properties for a single fluid phase
        #
        #
        # Input
        #       fphase - fluid phase; Oil, Water or Gas
        #       press  - fluid pressure value (MPa)
        #
        # Output
        #       pbulk - bulk modulus of fluid phase
        #               "fphase" for pressure value
        #               "press" (MPa)
        #
        # -----------------------------------------------
        #
        if fphase.lower() == "oil": # refers to water in Smeaheia
            press_range = np.array([0.10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
            # Bo values assume Rs = 0
            Bo_values = np.array(
                [1.00469, 1.00430, 1.00387, 1.00345, 1.00302, 1.00260, 1.00218, 1.00176, 1.00134, 1.00092, 1.00050,
                 1.00008, 0.99967, 0.99925, 0.99884, 0.99843, 0.99802, 0.99761, 0.99720, 0.99679, 0.99638])


        elif fphase.lower() == "gas":
            # Values from .DATA file for Smeaheia (converted to MPa)
            press_range = np.array(
                [0.101, 0.885, 1.669, 2.453, 3.238, 4.022, 4.806, 5.590, 6.2098, 7.0899, 7.6765, 8.2630, 8.8495, 9.4359,
                 10.0222, 10.6084, 11.1945, 14.7087, 17.6334, 20.856, 23.4695, 27.5419])  # Example pressures in MPa
            Bo_values = np.array(
                [1.07365, 0.11758, 0.05962, 0.03863, 0.02773, 0.02100, 0.01639, 0.01298, 0.010286, 0.007578, 0.005521,
                 0.003314, 0.003034, 0.002919, 0.002851, 0.002802, 0.002766, 0.002648, 0.002599, 0.002566, 0.002546,
                 0.002525])  # Example formation volume factors in m^3/kg

        # Calculate numerical derivative of Bo with respect to Pressure
        dBo_dP = - np.gradient(Bo_values, press_range)
        # Calculate isothermal compressibility (van der Waals)
        compressibility = (1 / Bo_values) * dBo_dP  # Resulting array of compressibility values
        bulk_mod = 1 / compressibility

        #
        # Find the index of the closest pressure value in b
        closest_index = (np.abs(press_range - press)).argmin()

        # Extract the corresponding value from a
        pbulk = bulk_mod[closest_index]

        #
        return pbulk

    #

    def _phaseprops(self, fphase, press, Rs=None):
        #
        # Calculate properties for a single fluid phase
        #
        #
        # Input
        #       fphase - fluid phase; Oil, Water or Gas
        #       press  - fluid pressure value (MPa)
        #
        # Output
        #       pdens - phase density of fluid phase
        #               "fphase" for pressure value
        #               "press" (kg/m³)
        #       pbulk - bulk modulus of fluid phase
        #               "fphase" for pressure value
        #               "press" (MPa)
        #
        # -----------------------------------------------
        #
        if fphase.lower() == "oil":
            coeffsrho = np.array([0.8, 829.9])
            coeffsbulk = np.array([10.42, 995.79])
        elif fphase.lower() == "wat":
            coeffsrho = np.array([0.3, 1067.3])
            coeffsbulk = np.array([9.0, 2807.6])
        elif fphase.lower() == "gas":
            coeffsrho = np.array([4.7, 13.4])
            coeffsbulk = np.array([2.75, 0.0])
        else:
            print("\nError in phaseprops method")
            print("Illegal fluid phase name.")
            print("Legal fluid phase names are (case "
                  "insensitive): Oil, Wat, and Gas.")
            sys.exit(1)
        #
        # Assume simple linear pressure dependencies.
        # Coefficients are inferred from
        # plots in Batzle and Wang, Geophysics, 1992,
        # (where I set the temperature to be 100 degrees
        # Celsius, Note also that they give densities in
        # g/cc). The resulting straight lines do not fit
        # the data extremely well, but they should
        # be sufficiently accurate for the purpose of
        # this project.
        #
        pdens = coeffsrho[0]*press + coeffsrho[1]
        pbulk = coeffsbulk[0]*press + coeffsbulk[1]
        #
        return pdens, pbulk

    #
    # =======================
    # Fluid properties end
    # =======================
    #

    #
    # =========================
    # Solid properties start
    # =========================
    #
    def _solidprops_Johansen(self):
        #
        # Calculate bulk and shear solid rock (mineral)
        # moduli by averaging Hashin-Shtrikman bounds
        #
        #
        # Input
        #       poro -porosity
        #
        # Output
        #       denss  - solid rock density (kg/m³)
        #       bulks  - solid rock bulk modulus (unit MPa)
        #       shears - solid rock shear modulus (unit MPa)
        #
        # -----------------------------------------------
        #
        #
        # Solid rock (mineral) density. (Note
        # that this is often termed \rho_dry, and not
        # \rho_s)

        denss = 2650 # Density of mineral/solid rock kg/m3

        #
        bulks = 37 # (GPa)
        shears = 44 # (GPa)
        bulks *= 1000  # Convert from GPa to MPa
        shears *= 1000
        #
        return denss, bulks, shears
    #
    #
    # =======================
    # Solid properties end
    # =======================
    #
    def _coordination_number(self):
        # Applies for granular media
        # Average number of contacts that each grain has with surrounding grains
        # Coordnumber = 6; simple cubic packing
        # Coordnumber = 12; hexagonal close packing
        # Needed for Hertz-Mindlin model
        # Smeaheia number (Tuhin)
        coordnumber = 9

        return coordnumber
    #
    def _critical_porosity(self):
        # For most porous media there exists a critical  porosity
        # phi_critical, that seperates their mechanical and acoustic behaviour into two domains.
        # For porosities below phi_critical the mineral grains are oad bearing, for values above the grains are
        # suspended in the fluids which are load-bearing
        # Needed for Hertz-Mindlin model
        # Smeaheia number (Tuhin)
        phicritical = 0.36

        return phicritical
    #
    def _effective_pressure(self, poverb, pfluid):

        # Input
        #       poverb - overburden pressure (MPa)
        #       pfluid - fluid pressure (MPa)

        peff = poverb - pfluid

        if peff < 0:
            print("\nError in _hertzmindlin method")
            print("Negative effective pressure (" + str(peff) +
              "). Setting effective pressure to 0.01")
            peff = 0.01



        return peff

    # ============================
    # Dry rock properties start
    # ============================
    #
    def _dryrockmoduli_Smeaheia(self, coordnumber, phicritical, poro, peff, bulks, shears):
        #
        #
        # Calculate bulk and shear dry rock moduli,

        #
        # Input
        #       poro   - porosity
        #       peff   - effective pressure overburden - fluid pressure (MPa)
        #       bulks  - bulk solid (mineral) rock bulk
        #                modulus (MPa)
        #       shears - solid rock (mineral) shear
        #                modulus (MPa)
        #
        # Output
        #       bulkd  - dry rock bulk modulus (unit
        #                inherited from hertzmindlin and
        #                variable; bulks)
        #       sheard - dry rock shear modulus (unit
        #                inherited from hertzmindlin and
        #                variable; shears)
        #
        # -----------------------------------------------
        #
        # Calculate Hertz-Mindlin moduli
        #
        bulkhm, shearhm = self._hertzmindlin_Mavko(peff, bulks, shears, coordnumber, phicritical)
        #
        bulkd = 1 / ((poro / phicritical) / (bulkhm + 4 / 3 * shearhm) +
                     (1 - poro / phicritical) / (bulks + 4 / 3 * shearhm)) - 4 / 3 * shearhm

        psi = (9 * bulkhm + 8 * shearhm) / (bulkhm + 2 * shearhm)

        sheard = 1 / ((poro / phicritical) / (shearhm + 1 / 6 * psi * shearhm) +
                     (1 - poro / phicritical) / (shears + 1 / 6 * psi * shearhm)) - 1 / 6 * psi * shearhm

        #return K_dry, G_dry
        return bulkd, sheard


            #
    # ---------------------------------------------------
    #

    def _hertzmindlin_Mavko(self, peff, bulks, shears, coordnumber, phicritical):
        #
        # Calculate bulk and shear Hertz-Mindlin moduli
        # adapted from Tuhins kode and "The rock physics handbook", pp247
        #
        #
        # Input
        #       p_eff       - effective pressure
        #       bulks       - bulk solid (mineral) rock bulk
        #                     modulus (MPa)
        #       shears      - solid rock (mineral) shear
        #                     modulus (MPa)
        #       coordnumber - average number of contacts that each grain has with surrounding grains
        #       phicritical - critical porosity
        #
        # Output
        #       bulkhm  - Hertz-Mindlin bulk modulus
        #                 (MPa)
        #       shearhm - Hertz-Mindlin shear modulus
        #                 (MPa)
        #
        # -----------------------------------------------
        #


        poisson = (3 * bulks - 2 * shears) / (6 * bulks + 2 * shears)

        bulkhm = ((coordnumber ** 2 * (1 - phicritical) ** 2 * shears ** 2 * peff) /
                (18 * np.pi ** 2 * (1 - poisson) ** 2)) ** (1 / 3)
        shearhm = (5 - 4 * poisson) / (10 - 5 * poisson) * \
               ((3 * coordnumber ** 2 * (1 - phicritical) ** 2 * shears ** 2 * peff) /
                (2 * np.pi ** 2 * (1 - poisson) ** 2)) ** (1 / 3)



        #
        return bulkhm, shearhm


    # ===========================
    # Dry rock properties end
    # ===========================


if __name__ == '__main__':
    #
    # Example input with two phases and three grid cells
    #
    porosity = [0.34999999, 0.34999999, 0.34999999]
#    pressure = [ 29.29150963, 29.14003944, 28.88845444]
    pressure = [29.3558, 29.2625, 29.3558]
#    pressure = [ 25.0, 25.0, 25.0]
    phases = ["Oil", "Wat"]
#    saturations = [[0.72783828, 0.66568458, 0.58033288],
#                   [0.27216172, 0.33431542, 0.41966712]]
    saturations = [[0.6358, 0.5755, 0.6358],
                   [0.3641, 0.4245, 0.3641]]
#    saturations = [[0.4, 0.5, 0.6],
#                   [0.6, 0.5, 0.4]]
    petElProp = "bulk velocity"
    input_dict = {}
    input_dict['overburden'] = 'overb.npz'

    print("\nInput:")
    print("porosity, pressure:", porosity, pressure)
    print("phases, saturations:", phases, saturations)
    print("petElProp:", petElProp)
    print("input_dict:", input_dict)

    satrock = elasticproperties(input_dict)

    print("overburden:", satrock.overburden)

    satrock.calc_props(phases, saturations, pressure,
                       porosity)

    print("\nOutput from calc_props:")
    print("Density:", satrock.getDens())
    print("Bulk modulus:", satrock.getBulkMod())
    print("Shear modulus:", satrock.getShearMod())
    print("Bulk velocity:", satrock.getBulkVel())
    print("Shear velocity:", satrock.getShearVel())
    print("Bulk impedance:", satrock.getBulkImp())
    print("Shear impedance:", satrock.getShearImp())

    satrock.getMatchProp(petElProp)

    print("\nOutput from getMatchProp:")
    print("Model output selected for data match:",
          satrock.match_prop)
