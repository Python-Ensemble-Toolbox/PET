"""Descriptive description."""

__author__ = 'TM'

# standardrp.py
import numpy as np
import sys
import multiprocessing as mp
# internal load
from misc.system_tools.environ_var import OpenBlasSingleThread  # Single threaded OpenBLAS runs


class elasticproperties:
    """
    Calculate elastic properties from standard
    rock-physics models, specifically following Batzle
    and Wang, Geophysics, 1992, for fluid properties, and
    Report 1 in Abul Fahimuddin's thesis at Universty of
    Bergen (2010) for other properties.

    Examples
    --------
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
        if 'baseline' in self.input_dict:
            self.baseline = self.input_dict['baseline']  # 4D baseline
        if 'parallel' in self.input_dict:
            self.parallel = self.input_dict['parallel']

    def _filter(self):
        bulkmod = self.bulkimp
        self.bulkimp = bulkmod.flatten()

    def setup_fwd_run(self, state):
        """
        Setup the input parameters to be used in the PEM simulator. Parameters can be a an ensemble or a single array.
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
                   porosity, wait_for_proc=None, ntg=None, Rs=None, press_init=None, ensembleMember=None):
        ###
        # This doesn't initialize for models with no uncertainty
        ###
        # # if some PEM properties have uncertainty, set the correct value
        # if ensembleMember is not None:
        #     for pem_state in self.__dict__.keys(): # loop over all possible pem vaules
        #         if pem_state is not 'inv_state': # do not alter the ensemble
        #             if type(eval('self.{}'.format(pem_state))) is dict:
        #                 for el in eval('self.{}'.format(pem_state)).keys():
        #                     if type(eval('self.{}'.format(pem_state))[el]) is dict:
        #                         for param in eval('self.{}'.format(pem_state))[el].keys():
        #                             if param in self.inv_state:
        #                                 eval('self.{}'.format(pem_state))[el][param]=\
        #                                     self.inv_state[param][:, ensembleMember]
        #                             elif param + '_' + el in self.inv_state:
        #                                 eval('self.{}'.format(pem_state))[el][param] = \
        #                                     self.inv_state[param+'_' + el][:, ensembleMember]
        #                     else:
        #                         if el in self.inv_state:
        #                             eval('self.{}'.format(pem_state))[el] = self.inv_state[el][:,ensembleMember]
        #             else:
        #                 if pem_state in self.inv_state:
        #                     setattr(self,pem_state, self.inv_state[pem_state][:,ensembleMember])

        # Check if the inputs are given as a list (more
        # than one phase) or a single input (single
        # phase). If single phase input, make the input a
        # list with a single entry (s.t. it can be used
        # directly with the methods below)
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
        # Load "overburden" into local variable to
        # comply with remaining code parts
        overburden = self.overburden
        if press_init is None:
            p_init = self.p_init
        else:
            p_init = press_init
        #
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

        for i in range(len(saturations[:, 0])):
            #
            # Calculate fluid properties
            #
            # set Rs if needed
            densf, bulkf = \
                self._fluidprops(self.phases,
                                 saturations[i, :], pressure[i], Rs[i])
            #
            denss, bulks, shears = \
                self._solidprops(porosity[i], ntg[i], i)
            #
            # Calculate dry rock moduli
            #

            bulkd, sheard = \
                self._dryrockmoduli(porosity[i],
                                    overburden[i],
                                    pressure[i], bulks,
                                    shears, i, ntg[i], p_init[i], denss, Rs[i], self.phases)
            # -------------------------------
            # Calculate saturated properties
            # -------------------------------
            #
            # Density
            #
            self.dens[i] = (porosity[i]*densf +
                            (1-porosity[i])*denss)*0.001
            #
            # Moduli
            #
            self.bulkmod[i] = \
                bulkd + (1 - bulkd/bulks)**2 / \
                (porosity[i]/bulkf +
                 (1-porosity[i])/bulks -
                 bulkd/(bulks**2))
            self.shearmod[i] = sheard

            # Velocities (due to bulk/shear modulus being
            # in MPa, we multiply by 1000 to get m/s
            # instead of km/s)
            #
            self.bulkvel[i] = \
                100*np.sqrt((abs(self.bulkmod[i]) +
                             4*self.shearmod[i]/3)/(self.dens[i]))
            self.shearvel[i] = \
                100*np.sqrt(self.shearmod[i] /
                            (self.dens[i]))
            #
            # Impedances (m/s)*(Kg/m3)
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

    #
    # ===================================================
    # Fluid properties start
    # ===================================================
    #
    def _fluidprops(self, fphases, fsats, fpress, Rs=None):
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
    def _solidprops(self, poro, ntg=None, ind=None):
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
        #       bulks  - solid rock bulk modulus (unit
        #                inherited from hashinshtr)
        #       shears - solid rock shear modulus (unit
        #                inherited from hashinshtr)
        #
        # -----------------------------------------------
        #
        # From PetroWiki (kg/m³)
        #
        densc = 2540
        densq = 2650
        #
        # From "Step 1" of "recipe" in Report 1 in Abul
        # Fahimuddin's thesis.
        #
        vclay = 0.7 - 1.58*poro
        #
        # Calculate solid rock (mineral) density. (Note
        # that this is often termed \rho_dry, and not
        # \rho_s)
        #
        denss = densq + vclay*(densc - densq)
        #
        # Calculate lower and upper bulk and shear
        # Hashin-Shtrikman bounds
        #
        bulkl, bulku, shearl, shearu = \
            self._hashinshtr(vclay)
        #
        # Calculate bulk and shear solid rock (mineral)
        # moduli as arithmetic means of the respective
        # bounds
        #
        bulkb = np.array([bulkl, bulku])
        shearb = np.array([shearl, shearu])
        bulks = np.mean(bulkb)
        shears = np.mean(shearb)
        #
        return denss, bulks, shears
    #
    # ---------------------------------------------------
    #

    def _hashinshtr(self, vclay):
        #
        # Calculate lower and upper, bulk and shear,
        # Hashin-Shtrikman bounds, utilizing that they
        # all have the common mathematical form,
        #
        # f = a + b/((1/c) + d*(1/e)).
        #
        #
        # Input
        #       vclay - "volume of clay"
        #
        # Output
        #       bulkl  - lower bulk Hashin-Shtrikman
        #                bound (MPa)
        #       bulku  - upper bulk Hashin-Shtrikman
        #                bound (MPa)
        #       shearl - lower shear Hashin-Shtrikman
        #                bound (MPa)
        #       shearu - upper shear Hashin-Shtrikman
        #                bound (MPa)
        #
        # -----------------------------------------------
        #
        # From table 1 in Report 1 in Abul Fahimuddin's
        # thesis (he used GPa, I use MPa), ("c" for clay
        # and "q" for quartz.):
        #
        bulkc = 14900
        bulkq = 37000
        shearc = 1950
        shearq = 44000
        #
        # Calculate quantities common for both bulk and
        # shear formulas
        #
        lb = 1 - vclay
        ub = vclay
        le = bulkc + 4*shearc/3
        ue = bulkq + 4*shearq/3
        #
        # Calculate quantities common only for bulk
        # formulas
        #
        bld = vclay
        bud = 1 - vclay
        blc = bulkq - bulkc
        buc = bulkc - bulkq
        #
        # Calculate quantities common only for shear
        # formulas
        #
        sld = 2*vclay*(bulkc + 2*shearc)/(5*shearc)
        sud = 2*(1 - vclay)*(bulkq + 2*shearq)/(5*shearq)
        slc = shearq - shearc
        suc = shearc - shearq
        #
        # Calculate bounds utilizing generic formula;
        # f = a + b/((1/c) + d*(1/e)).
        #
        # Lower bulk
        #
        bulkl = self._genhashinshtr(bulkc, lb, blc, bld,
                                    le)
        #
        # Upper bulk
        #
        bulku = self._genhashinshtr(bulkq, ub, buc, bud,
                                    ue)
        #
        # Lower shear
        #
        shearl = self._genhashinshtr(shearc, lb, slc,
                                     sld, le)
        #
        # Upper shear
        #
        shearu = self._genhashinshtr(shearq, ub, suc,
                                     sud, ue)
        #
        return bulkl, bulku, shearl, shearu

    #
    # ---------------------------------------------------
    #
    def _genhashinshtr(self, a, b, c, d, e):
        #
        # Calculate arbitrary Hashin-Shtrikman bound,
        # which has the generic form
        #
        # f = a + b/((1/c) + d*(1/e))
        #
        # both for lower and upper bulk and shear bounds
        #
        #
        # Input
        #       a - see above formula
        #       b - see above formula
        #       c - see above formula
        #       d - see above formula
        #       e - see above formula
        #
        # Output
        #       f - Bulk or shear Hashin-Shtrikman bound
        #           value
        #
        # -----------------------------------------------
        #
        cinv = 1/c
        einv = 1/e
        f = a + b/(cinv + d*einv)
        #
        return f
    #
    # =======================
    # Solid properties end
    # =======================
    #

    #
    # ============================
    # Dry rock properties start
    # ============================
    #
    def _dryrockmoduli(self, poro, poverb, pfluid, bulks, shears, ind=None, ntg=None, p_init=None, denss=None, Rs=None, phases=None):
        #
        #
        # Calculate bulk and shear dry rock moduli,
        # utilizing that they have the common
        # mathematical form,
        #
        #     --                                -- ^(-1)
        #     |(poro/poroc)      1 - (poro/poroc)|
        # f = |------------   +  ----------------|   - z.
        #     |   a + z              b + z       |
        #     --.                               --
        #
        #
        # Input
        #       poro   - porosity
        #       poverb - overburden pressure (MPa)
        #       pfluid - fluid pressure (MPa)
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
        bulkhm, shearhm = self._hertzmindlin(poverb,
                                             pfluid)
        #
        # From table 1 in Report 1 in Abul Fahimuddin's
        # thesis (I assume \phi_max in that
        # table corresponds to \phi_c in his formulas):
        #
        poroc = 0.4
        #
        # Calculate input common to both bulk and
        # shear formulas
        #
        poratio = poro/poroc
        #
        # Calculate input to bulk formula
        #
        ba = bulkhm
        bb = bulks
        bz = 4*shearhm/3
        #
        # Calculate input to shear formula
        #
        sa = shearhm
        sb = shears
        sz = (shearhm/6)*((9*bulkhm + 8*shearhm) /
                          (bulkhm + 2*shearhm))
        #
        # Calculate moduli
        #
        bulkd = self._gendryrock(poratio, ba, bb, bz)
        sheard = self._gendryrock(poratio, sa, sb, sz)
        #
        return bulkd, sheard
    #
    # ---------------------------------------------------
    #

    def _hertzmindlin(self, poverb, pfluid):
        #
        # Calculate bulk and shear Hertz-Mindlin moduli
        # utilizing that they have the common
        # mathematical form,
        #
        # f = <a>max*(peff/pref)^kappa.
        #
        #
        # Input
        #       poverb - overburden pressure
        #       pfluid - fluid pressure
        #
        # Output
        #       bulkhm  - Hertz-Mindlin bulk modulus
        #                 (MPa)
        #       shearhm - Hertz-Mindlin shear modulus
        #                 (MPa)
        #
        # -----------------------------------------------
        #
        # From table 1 in Report 1 in Abul Fahimuddin's
        # thesis (he used GPa for the moduli, I use MPa
        # also for them):
        #
        bulkmax = 3310
        shearmax = 2840
        pref = 8.8
        kappa = 0.233
        #
        # Calculate moduli
        #
        peff = poverb - pfluid
        if peff < 0:
            print("\nError in _hertzmindlin method")
            print("Negative effective pressure (" + str(peff) +
                  "). Setting effective pressure to 0.01")
            peff = 0.01
   #         sys.exit(1)
        common = (peff/pref)**kappa
        bulkhm = bulkmax*common
        shearhm = shearmax*common
        #
        return bulkhm, shearhm
    #
    # ---------------------------------------------------
    #

    def _gendryrock(self, q, a, b, z):
        #
        # Calculate arbitrary dry rock moduli, which has
        # the generic form
        #
        #     --                                -- ^(-1)
        #     |     q                1 - q       |
        # f = |------------   +  ----------------|   - z,
        #     |   a + z              b + z       |
        #     --.                               --
        #
        # both for bulk and shear moduli
        #
        #
        # Input
        #       q - see above formula
        #       a - see above formula
        #       b - see above formula
        #       z - see above formula
        #
        # Output
        #       f - Bulk or shear dry rock modulus value
        #
        # -----------------------------------------------
        #
        afrac = q/(a + z)
        bfrac = (1 - q)/(b + z)
        f = 1/(afrac + bfrac) - z
        #
        return f
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
