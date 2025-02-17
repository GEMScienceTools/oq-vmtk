import numpy as np
from scipy.linalg import eigh
import openseespy.opensees as ops
import matplotlib.pyplot as plt
import os 

## Define plot style
HFONT = {'fontname':'Helvetica'}

FONTSIZE_1 = 16
FONTSIZE_2 = 14
FONTSIZE_3 = 12

LINEWIDTH_1= 3
LINEWIDTH_2= 2
LINEWIDTH_3 = 1

RESOLUTION = 500
MARKER_SIZE_1 = 100
MARKER_SIZE_2 = 60
MARKER_SIZE_3 = 10

GEM_COLORS  = ["#0A4F4E","#0A4F5E","#54D7EB","#54D6EB","#399283","#399264","#399296"]

class modeller():

    """
    Details
    -------
    Class of functions to model and analyse multi-degree-of-freedom oscillators
    """
    
    def __init__(self, number_storeys, floor_heights, floor_masses, storey_disps, storey_forces, degradation):
        """
        ----------
        Parameters
        ----------
        number_storeys:            int                Number of storeys
        floor_heights:            list                Floor heights in metres (e.g. [2.5, 3.0])
        floor_masses:             list                Floor masses in tonnes (e.g. [1000, 1200])
        storey_disps:            array                Storey displacements (size = nst, CapPoints)
        storey_forces:           array                Storey forces (size = nst,CapPoints)
        degradation:              bool                Set True/False to active hysteresis
        """

        ### Run tests on input parameters
        if len(floor_heights)!=number_storeys or len(floor_masses)!=number_storeys:
            raise ValueError('Number of entries exceed the number of storeys!')
        
        self.number_storeys = number_storeys
        self.floor_heights  = floor_heights
        self.floor_masses   = floor_masses
        self.storey_disps   = storey_disps 
        self.storey_forces  = storey_forces
        self.degradation    = degradation
 

    def create_Pinching4_material(self, mat1Tag, mat2Tag, storey_forces, storey_disps, degradation):   
        """
        Function to create Pinching4 Material Model used for the mdof_material object of stickModel
        -----
        Input
        -----
        :param mat1Tag:           int                Material Tag #1
        :param mat2Tag:           int                Material Tag #2
        :param F:               array                Array of storey forces
        :param D:               array                Array of storey displacements
        :param degradation:      bool                Boolean condition to enable/disable degradation in Pinching4
    
        -----
        Output
        -----
        None
     
        """
    
        force=np.zeros([5,1])
        disp =np.zeros([5,1])
        
        # Bilinear
        if len(storey_forces)==2:
              #bilinear curve
              force[1]=storey_forces[0]
              force[4]=storey_forces[-1]
              
              disp[1]=storey_disps[0]
              disp[4]=storey_disps[-1]
              
              disp[2]=disp[1]+(disp[4]-disp[1])/3
              disp[3]=disp[1]+2*((disp[4]-disp[1])/3)
              
              force[2]=np.interp(disp[2],storey_disps,storey_forces)
              force[3]=np.interp(disp[3],storey_disps,storey_forces)
        
        # Trilinear
        elif len(storey_forces)==3:
              
              force[1]=storey_forces[0]
              force[4]=storey_forces[-1]
              
              disp[1]=storey_disps[0]
              disp[4]=storey_disps[-1]
              
              force[2]=storey_forces[1]
              disp[2] =storey_disps[1]
              
              disp[3]=np.mean([disp[2],disp[-1]])
              force[3]=np.interp(disp[3],storey_disps,storey_forces)
        
        # Quadrilinear
        elif len(storey_forces)==4:
              force[1]=storey_forces[0]
              force[4]=storey_forces[-1]
              
              disp[1]=storey_disps[0]
              disp[4]=storey_disps[-1]
              
              force[2]=storey_forces[1]
              disp[2]=storey_disps[1]
              
              force[3]=storey_forces[2]
              disp[3]=storey_disps[2]
    
        if degradation==True:
            matargs=[force[1,0],disp[1,0],force[2,0],disp[2,0],force[3,0],disp[3,0],force[4,0],disp[4,0],
                                 -1*force[1,0],-1*disp[1,0],-1*force[2,0],-1*disp[2,0],-1*force[3,0],-1*disp[3,0],-1*force[4,0],-1*disp[4,0],
                                 0.5,0.25,0.05,
                                 0.5,0.25,0.05,
                                 0,0.1,0,0,0.2,
                                 0,0.1,0,0,0.2,
                                 0,0.4,0,0.4,0.9,
                                 10,'energy']
        else:   
            matargs=[force[1,0],disp[1,0],force[2,0],disp[2,0],force[3,0],disp[3,0],force[4,0],disp[4,0],
                                 -1*force[1,0],-1*disp[1,0],-1*force[2,0],-1*disp[2,0],-1*force[3,0],-1*disp[3,0],-1*force[4,0],-1*disp[4,0],
                                 0.5,0.25,0.05,
                                 0.5,0.25,0.05,
                                 0,0,0,0,0,
                                 0,0,0,0,0,
                                 0,0,0,0,0,
                                 10,'energy']
        
        ops.uniaxialMaterial('Pinching4', mat1Tag,*matargs)
        ops.uniaxialMaterial('MinMax', mat2Tag, mat1Tag, '-min', -1*disp[4,0], '-max', disp[4,0])

    def compile_model(self):
        """
        Compiles the multi-degree-of-freedom oscillator

        Parameters
        ----------
        None.
        
        Returns
        -------
        None.

        """    
        ### Set model builder
        ops.wipe() # wipe existing model
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        
        ### Define base node (tag = 0)
        ops.node(0, *[0.0, 0.0, 0.0])
        ### Define floor nodes (tag = 1+)
        i = 1
        current_height = 0.0
        while i <= self.number_storeys:
            nodeTag = i
            current_height = current_height + self.floor_heights[i-1]
            current_mass = self.floor_masses[i-1]
            coords = [0.0, 0.0, current_height]
            masses = [current_mass, current_mass, 1e-6, 1e-6, 1e-6, 1e-6]
            ops.node(nodeTag,*coords)
            ops.mass(nodeTag,*masses)
            i+=1
        
        ### Get list of model nodes
        nodeList = ops.getNodeTags()
        ### Impose boundary conditions
        for i in nodeList:
            # fix the base node against all DOFs
            if i==0:
                ops.fix(i,1,1,1,1,1,1)
            # release the horizontal DOFs (1,2) and fix remaining
            else:
                ops.fix(i,0,0,1,1,1,1)
         
        ### Get number of zerolength elements required
        nodeList = ops.getNodeTags()
        numEle = len(nodeList)-1
                        
        for i in range(self.number_storeys):
            
            ### define the material tag associated with each storey
            mat1Tag = int(f'1{i}00') # hysteretic material tag
            mat2Tag = int(f'1{i}01') # min-max material tag
            
            ### get the backbone curve definition
            current_storey_disps = self.storey_disps[i,:].tolist() # deformation capacity (i.e., storey displacement in m)
            current_storey_forces = self.storey_forces[i,:].tolist() # strength capacity (i.e., storey base shear in kN)
                        
            ### Create rigid elastic materials for the restrained dofs
            rigM = int(f'1{i}02')
            ops.uniaxialMaterial('Elastic', rigM, 1e16)
                        
            ### Create the nonlinear material for the unrestrained dofs
            self.create_Pinching4_material(mat1Tag, mat2Tag, current_storey_forces, current_storey_disps, self.degradation)
            
            ### Aggregate materials
            matTags = [mat2Tag, rigM, rigM, rigM, rigM, rigM]            

            ### Define element connectivity
            eleTag = int(f'200{i}')
            eleNodes = [i, i+1]
            
            ### Create the element
            ops.element('zeroLength', eleTag, eleNodes[0], eleNodes[1], '-mat', mat2Tag, mat2Tag, rigM, rigM, rigM, rigM, '-dir', 1, 2, 3, 4, 5, 6, '-doRayleigh', 1)
       

    def plot_model(self, display_info=True):
        """
        Plots the Opensees model
    
        Parameters
        ----------
        None.
        
        Returns
        -------
        None.
    
        """

        modelLineColor = 'blue'
        modellinewidth = 1
        Vert = 'Z'
               
        # get list of model nodes
        NodeCoordListX = []; NodeCoordListY = []; NodeCoordListZ = [];
        NodeMassList = []
        
        nodeList = ops.getNodeTags()
        for thisNodeTag in nodeList:
            NodeCoordListX.append(ops.nodeCoord(thisNodeTag,1))
            NodeCoordListY.append(ops.nodeCoord(thisNodeTag,2))
            NodeCoordListZ.append(ops.nodeCoord(thisNodeTag,3))
            NodeMassList.append(ops.nodeMass(thisNodeTag,1))
        
        # get list of model elements
        elementList = ops.getEleTags()
        for thisEleTag in elementList:
            eleNodesList = ops.eleNodes(thisEleTag)
            if len(eleNodesList)==2:
                [NodeItag,NodeJtag] = eleNodesList
                NodeCoordListI=ops.nodeCoord(NodeItag)
                NodeCoordListJ=ops.nodeCoord(NodeJtag)
                [NodeIxcoord,NodeIycoord,NodeIzcoord]=NodeCoordListI
                [NodeJxcoord,NodeJycoord,NodeJzcoord]=NodeCoordListJ
        
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(projection='3d')
        
        for i in range(len(nodeList)):
            ax.scatter(NodeCoordListX[i],NodeCoordListY[i],NodeCoordListZ[i],s=MARKER_SIZE_2,color=GEM_COLORS[0])
            if display_info == True:
                ax.text(NodeCoordListX[i],NodeCoordListY[i],NodeCoordListZ[i],  'Node %s (%s,%s,%s)' % (str(i),str(NodeCoordListX[i]),str(NodeCoordListY[i]),str(NodeCoordListZ[i])), size=20, zorder=1, color=GEM_COLORS[1]) 
        
        i = 0
        while i < len(elementList):
            
            x = [NodeCoordListX[i], NodeCoordListX[i+1]]
            y = [NodeCoordListY[i], NodeCoordListY[i+1]]
            z = [NodeCoordListZ[i], NodeCoordListZ[i+1]]
            
            plt.plot(x,y,z,color='blue')
            i = i+1
        
        ax.set_xlabel('X-Direction [m]', fontsize=FONTSIZE_2)
        ax.set_ylabel('Y-Direction [m]', fontsize=FONTSIZE_2)
        ax.set_zlabel('Z-Direction [m]', fontsize=FONTSIZE_2)
        
        plt.show()

##########################################################################
#                             ANALYSIS MODULES                           #
##########################################################################
    def do_gravity_analysis(self, nG=100,ansys_soe='UmfPack',constraints_handler='Transformation',
                            numberer='RCM',test_type='NormDispIncr',init_tol = 1.0e-6, init_iter = 500, 
                            algorithm_type='Newton' , integrator='LoadControl',analysis='Static'):        
        """
        Perform gravity analysis on MDOF
    
        Parameters
        ----------
        nG:                             int                Number of gravity analysis steps to perform.
        ansys_soe:                   string                System of equations type.
        constraints_handler:         string                The constraints handler object determines how the constraint equations are enforced in the analysis. Constraint equations enforce a specified value for a DOF, or a relationship between DOFs.
        numberer:                    string                The DOF numberer object determines the mapping between equation numbers and degrees-of-freedom – how degrees-of-freedom are numbered.
        test_type:                   string                This command is used to construct the LinearSOE and LinearSolver objects to store and solve the test of equations in the analysis.
        init_tol:                     float                Tolerance criteria used to check for convergence.
        init_iter:                    float                Max number of iterations to check.
        algorithm_type:              string                The integrator object determines the meaning of the terms in the system of equation object Ax=B.
        analysis:                    string                The analysis object, which defines what type of analysis is to be performed.
        
        Returns
        -------
        None.
    
        """
        
        ### Define the analysis objects and run gravity analysis
        ops.system(ansys_soe) # creates the system of equations, a sparse solver with partial pivoting
        ops.constraints(constraints_handler) # creates the constraint handler, the transformation method
        ops.numberer(numberer) # creates the DOF numberer, the reverse Cuthill-McKee algorithm
        ops.test(test_type, init_tol, init_iter, 3) # creates the convergence test
        ops.algorithm(algorithm_type) # creates the solution algorithm, a Newton-Raphson algorithm
        ops.integrator(integrator, (1/nG)) # creates the integration scheme
        ops.analysis(analysis) # creates the analysis object
        ops.analyze(nG) # perform the gravity load analysis
        ops.loadConst('-time', 0.0)

        ### Wipe the analysis objects
        ops.wipeAnalysis()
        
    def do_modal_analysis(self, num_modes=3, solver = '-genBandArpack', doRayleigh=False, pflag=False):
        """
        Perform modal analysis on MDOF
    
        Parameters
        ----------
        num_modes:                      int                Number of modes to consider (default is 3).
        solver:                      string                Type of solver (default is -genBandArpack).
        doRayleigh:                    bool                Flag to enable/disable Rayleigh damping
        pflag:                         bool                Flag to print (or not) the modal analysis report.
        
        Returns
        -------
        T:                            array                Periods of vibration.
        mode_shape                     list                First mode-based normalised mode-shape
        """        
    
        ### Get frequency and period        
        self.omega = np.power(ops.eigen(solver, num_modes), 0.5)
        T = 2.0*np.pi/self.omega
        ### Get list of model nodes
        nodeList = ops.getNodeTags() 

        mode_shape = []        
        # Extract mode shapes for all nodes (displacements in x)
        for k in range(1, self.number_storeys+1):
            ux = ops.nodeEigenvector(k, 1, 1)  # Displacement in x-direction
            mode_shape.append(ux)
        
        # Normalize the mode shape
        mode_shape = np.array(mode_shape)/mode_shape[-1]
           
        ### Print optional report
        if pflag:
            ops.modalProperties('-print')
            ### Print output
            print(r'Fundamental Period:  T = {:.3f} s'.format(T[0]))
            print('Mode Shape:', mode_shape)
            
        ### Wipe the analysis objects
        ops.wipeAnalysis()      
        
        return T, mode_shape
            
    def do_spo_analysis(self, ref_disp, disp_scale_factor, push_dir, phi, pflag=True, 
                        num_steps=200, ansys_soe='BandGeneral', constraints_handler='Transformation', 
                        numberer='RCM', test_type='EnergyIncr', init_tol=1.0e-5, init_iter=1000, 
                        algorithm_type='KrylovNewton'):
        """
        Perform static pushover analysis on MDOF
    
        Parameters
        ----------
        ref_disp:                     float                Reference displacement to analyses are run. Corresponds to yield or equivalent other, such as 1mm.
        disp_scale_factor:            float                Multiple of ref_disp to which the push is run. So pushover can be run to a specified ductility or displacement.
        push_dir:                       int                Direction of pushover (1 = X; 2 = Y; 3 = Z)
        phi:                           list                Shape of lateral load applied, this is one (optional) output of the calibrateModel function and using it here assumes the applied pushover load is first-mode based
        pflag:                         bool                Flag to print (or not) the static pushover analysis steps
        num_steps:                      int                Number of spo analysis steps to perform (Default is 200).
        ansys_soe:                   string                System of equations type (Default is 'BandGeneral').
        constraints_handler:         string                The constraints handler object determines how the constraint equations are enforced in the analysis. Constraint equations enforce a specified value for a DOF, or a relationship between DOFs (Default is 'Transformation').
        numberer:                    string                The DOF numberer object determines the mapping between equation numbers and degrees-of-freedom – how degrees-of-freedom are numbered (Default is 'RCM').
        test_type:                   string                This command is used to construct the LinearSOE and LinearSolver objects to store and solve the test of equations in the analysis (Default is 'EnergyIncr').
        init_tol:                     float                Tolerance criteria used to check for convergence (Default is 1e-5).
        init_iter:                    float                Max number of iterations to check (Default is 1000).
        algorithm_type:              string                The integrator object determines the meaning of the terms in the system of equation object Ax=B (Default is 'KrylovNewton').
        
        Returns
        -------
        spo_disps:                    array                Displacements at each floor
        spo_rxn:                      array                Base shear as the sum of the reaction at the base 
        spo_disps_spring:             array                Displacements in the storey zero-length elements
        spo_forces_spring:            array                Shear forces in the storey zero-length elements
        """        

        # apply the load pattern
        ops.timeSeries("Linear", 1) # create timeSeries
        ops.pattern("Plain", 1, 1) # create a plain load pattern
        
        # define control nodes
        nodeList = ops.getNodeTags()
        control_node = nodeList[-1]
        pattern_nodes = nodeList[1:]
        rxn_nodes = [nodeList[0]]
        
                
        # we can integrate modal patterns, inverse triangular, etc.
        for i in np.arange(len(pattern_nodes)):
            if push_dir == 1:    
                if len(pattern_nodes)==1:
                    ops.load(pattern_nodes[i], 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                else:
                    ops.load(pattern_nodes[i], phi[i]*self.floor_masses[i], 0.0, 0.0, 0.0, 0.0, 0.0) ######### IT STARTS FROM ZERO

                    
                    # if len(pattern_nodes) <= 4:
                    #     ops.load(pattern_nodes[i], nodeList[i+1]/len(pattern_nodes), 0.0, 0.0, 0.0, 0.0, 0.0) ######### IT STARTS FROM ZERO
                    # elif len(pattern_nodes) >4:
                    #     ops.load(pattern_nodes[i], 4/3*nodeList[i+1]/len(pattern_nodes)*(1 - nodeList[i+1]/4/len(pattern_nodes)), 0.0, 0.0, 0.0, 0.0, 0.0) ######### IT STARTS FROM ZERO                        
            elif push_dir == 2:
                if len(pattern_nodes)==1:
                    ops.load(pattern_nodes[i], 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
                else:
                    ops.load(pattern_nodes[i], 0.0, phi[i]*self.floor_masses[i], 0.0, 0.0, 0.0, 0.0)
    
            elif push_dir == 3:
                if len(pattern_nodes)==1:
                    ops.load(pattern_nodes[i], 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
                else:
                    ops.load(pattern_nodes[i], 0.0, 0.0, nodeList[i]/len(pattern_nodes), 0.0, 0.0, 0.0)
                            
        # Set up the initial objects
        ops.system(ansys_soe)
        ops.constraints(constraints_handler)
        ops.numberer(numberer)
        ops.test(test_type, init_tol, init_iter)        
        ops.algorithm(algorithm_type)
        
        # Set the integrator
        target_disp = float(ref_disp)*float(disp_scale_factor)
        delta_disp = target_disp/(1.0*num_steps)
        ops.integrator('DisplacementControl', control_node, push_dir, delta_disp)
        ops.analysis('Static')
                
        # Get a list of all the element tags (zero-length springs)
        elementList = ops.getEleTags()
        
        # Give some feedback if requested
        if pflag is True:
            print(f"\n------ Static Pushover Analysis of Node # {control_node} to {target_disp} ---------")
        # Set up the analysis
        ok = 0
        step = 1
        loadf = 1.0
        
        # Recording base shear
        spo_rxn = np.array([0.])
        # Recording top displacement
        spo_top_disp = np.array([ops.nodeResponse(control_node, push_dir,1)])
        # Recording all displacements to estimate drifts
        spo_disps = np.array([[ops.nodeResponse(node, push_dir, 1) for node in pattern_nodes]])
        
        # Recording displacements and forces in non-linear zero-length springs [the zero is needed to get the exact required value]
        spo_disps_spring = np.array([[ops.eleResponse(ele, 'deformation')[0] for ele in elementList]])
        spo_forces_spring = np.array([[ops.eleResponse(ele, 'force')[0] for ele in elementList]])

        
        # Start the adaptive convergence scheme
        while step <= num_steps and ok == 0 and loadf > 0:
            
            # Push it by one step
            ok = ops.analyze(1)
    
            # If the analysis fails, try the following changes to achieve convergence
            if ok != 0:
                print('FAILED: Trying relaxing convergence...')
                ops.test(test_type, init_tol*0.01, init_iter)
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
            if ok != 0:
                print('FAILED: Trying relaxing convergence with more iterations...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
            if ok != 0:
                print('FAILED: Trying relaxing convergence with more iteration and Newton with initial then current...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ops.algorithm('Newton', 'initialThenCurrent')
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                print('FAILED: Trying relaxing convergence with more iteration and Newton with initial...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ops.algorithm('Newton', 'initial')
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                print('FAILED: Attempting a Hail Mary...')
                ops.test('FixedNumIter', init_iter*10)
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
            
            # This feature of disabling the possibility of having a negative loading has been included.
            loadf = ops.getTime()
                
            # Give some feedback if requested
            if pflag is True:
                curr_disp = ops.nodeDisp(control_node, push_dir)
                print('Currently pushed node ' + str(control_node) + ' to ' + str(curr_disp) + ' with ' + str(loadf))
                       
            # Increment to the next step
            step += 1
            
            # Get the results
            spo_top_disp = np.append(spo_top_disp, ops.nodeResponse(
            control_node, push_dir, 1))
            
            spo_disps = np.append(spo_disps, np.array([
            [ops.nodeResponse(node, push_dir, 1) for node in pattern_nodes]
            ]), axis=0)
    
           
            spo_disps_spring = np.append(spo_disps_spring, np.array([
            [ops.eleResponse(ele, 'deformation')[0] for ele in elementList]
            ]), axis=0)
            
            
            spo_forces_spring = np.append(spo_forces_spring, np.array([
            [ops.eleResponse(ele, 'force')[0] for ele in elementList]
            ]), axis=0)
               
            ops.reactions()
            temp = 0
            for n in rxn_nodes:
                temp += ops.nodeReaction(n, push_dir)
            spo_rxn = np.append(spo_rxn, -temp)
            
                
        # Give some feedback on what happened  
        if ok != 0:
            print('------ ANALYSIS FAILED --------')
        elif ok == 0:
            print('~~~~~~~ ANALYSIS SUCCESSFUL ~~~~~~~~~')        
        if loadf < 0:
            print('Stopped because of load factor below zero')
        
        ### Wipe the analysis objects
        ops.wipeAnalysis()
             
        return spo_disps, spo_rxn, spo_disps_spring, spo_forces_spring
    
    def do_cpo_analysis(self, ref_disp, mu, numCycles, push_dir, dispIncr, pflag=True, 
                        num_steps=200, ansys_soe='BandGeneral', constraints_handler='Transformation', 
                        numberer='RCM', test_type='NormDispIncr', init_tol=1.0e-5, init_iter=1000,
                        algorithm_type='KrylovNewton'):
        """
        Perform cyclic pushover analysis on MDOF
    
        Parameters
        ----------
        ref_disp:                     float                Reference displacement to analyses are run. Corresponds to yield or equivalent other, such as 1mm.
        mu:                           float                Target ductility.
        numCycles:                    float                Number of cycles.
        dispIncr:                     float                Number of displacement increments.
        push_dir:                       int                Direction of pushover (1 = X; 2 = Y; 3 = Z).
        pflag:                         bool                Flag to print (or not) the static pushover analysis steps (Default is 'True').
        num_steps:                      int                Number of cpo analysis steps to perform (Default is 200).
        ansys_soe:                   string                System of equations type. (Default is 'BandGeneral')
        constraints_handler:         string                The constraints handler object determines how the constraint equations are enforced in the analysis. Constraint equations enforce a specified value for a DOF, or a relationship between DOFs (Default is 'Transformation').
        numberer:                    string                The DOF numberer object determines the mapping between equation numbers and degrees-of-freedom – how degrees-of-freedom are numbered (Default is 'RCM').
        test_type:                   string                This command is used to construct the LinearSOE and LinearSolver objects to store and solve the test of equations in the analysis (Default is 'NormDispIncr').
        init_tol:                     float                Tolerance criteria used to check for convergence (Default is 1e-5).
        init_iter:                    float                Max number of iterations to check (Default is 1000).
        algorithm_type:              string                The integrator object determines the meaning of the terms in the system of equation object Ax=B (Default is 'KrylovNewton').
        
        Returns
        -------
        cpo_disps:                    array                Displacements at each floor.
        cpo_rxn:                      array                Base shear as the sum of the reaction at the base.
    
        """        

        # apply the load pattern
        ops.timeSeries("Linear", 1) # create timeSeries
        ops.pattern("Plain",1,1) # create a plain load pattern
                
        # define control nodes
        nodeList = ops.getNodeTags()
        control_node = nodeList[-1]
        pattern_nodes = nodeList[1:]
        rxn_nodes = [nodeList[0]]
        
        # we can integrate modal patterns, inverse triangular, etc.
        for i in np.arange(len(pattern_nodes)):
            if push_dir == 1:
                ops.load(pattern_nodes[i], nodeList[i]/len(pattern_nodes), 0.0, 0.0, 0.0, 0.0, 0.0)
            elif push_dir == 2:
                ops.load(pattern_nodes[i], 0.0, nodeList[i]/len(pattern_nodes), 0.0, 0.0, 0.0, 0.0)
            elif push_dir == 3:
                ops.load(pattern_nodes[i], 0.0, 0.0, nodeList[i]/len(pattern_nodes), 0.0, 0.0, 0.0)
        
        # Set up the initial objects
        ops.system(ansys_soe)
        ops.constraints(constraints_handler)
        ops.numberer(numberer)
        ops.test(test_type, init_tol, init_iter)        
        ops.algorithm(algorithm_type)

     	#create the list of displacements
        dispList =                  [ref_disp*mu, -2.0*ref_disp*mu, ref_disp*mu]
        cycleDispList = (numCycles* [ref_disp*mu, -2.0*ref_disp*mu, ref_disp*mu])
        dispNoMax = len(cycleDispList)
        
        # Give some feedback if requested
        if pflag is True:
            print(f"\n------ Cyclic Pushover Analysis of Node # {control_node} for {numCycles} cycles to ductility: {mu}---------")
        
        # Recording base shear
        cpo_rxn = np.array([0.])
        # Recording top displacement
        cpo_top_disp = np.array([ops.nodeResponse(control_node, push_dir,1)])
        # Recording all displacements to estimate drifts
        cpo_disps = np.array([[ops.nodeResponse(node, push_dir, 1) for node in pattern_nodes]])

    
        for d in range(dispNoMax):
            numIncr = dispIncr
            dU = cycleDispList[d]/(1.0*numIncr)
            ops.integrator('DisplacementControl', control_node, push_dir, dU)
            ops.analysis('Static')
            
            for l in range(numIncr):
                ok = ops.analyze(1)
                
                # If the analysis fails, try the following changes to achieve convergence
                if ok != 0:
                    print('FAILED: Trying relaxing convergence...')
                    ops.test(test_type, init_tol*0.01, init_iter)
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter)
                if ok != 0:
                    print('FAILED: Trying relaxing convergence with more iterations...')
                    ops.test(test_type, init_tol*0.01, init_iter*10)
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter)
                if ok != 0:
                    print('FAILED: Trying relaxing convergence with more iteration and Newton with initial then current...')
                    ops.test(test_type, init_tol*0.01, init_iter*10)
                    ops.algorithm('Newton', 'initialThenCurrent')
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter)
                    ops.algorithm(algorithm_type)
                if ok != 0:
                    print('FAILED: Trying relaxing convergence with more iteration and Newton with initial...')
                    ops.test(test_type, init_tol*0.01, init_iter*10)
                    ops.algorithm('Newton', 'initial')
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter)
                    ops.algorithm(algorithm_type)
                if ok != 0:
                    print('FAILED: Attempting a Hail Mary...')
                    ops.test('FixedNumIter', init_iter*10)
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter)
                if ok != 0:
                    print('Analysis Failed')
                    break 
                                
            # Give some feedback if requested
            if pflag is True:
                curr_disp = ops.nodeDisp(control_node, push_dir)
                print('Currently pushed node ' + str(control_node) + ' to ' + str(curr_disp))
                            
            # Get the results
            cpo_top_disp = np.append(cpo_top_disp, ops.nodeResponse(
            control_node, push_dir, 1))
    
            cpo_disps = np.append(cpo_disps, np.array([
            [ops.nodeResponse(node, push_dir, 1) for node in pattern_nodes]
            ]), axis=0)
                
            ops.reactions()
            temp = 0
            for n in rxn_nodes:
                temp += ops.nodeReaction(n, push_dir)
            cpo_rxn = np.append(cpo_rxn, -temp)
        
        ### Wipe the analysis objects
        ops.wipeAnalysis()
                      
        return cpo_disps, cpo_rxn
                    
    def do_nrha_analysis(self, fnames, dt_gm, sf, t_max, dt_ansys, nrha_outdir,
                         pflag=True, xi = 0.05, ansys_soe='BandGeneral', 
                         constraints_handler='Plain', numberer='RCM', 
                         test_type='NormDispIncr', init_tol=1.0e-6, init_iter=50, 
                         algorithm_type='Newton'):
        """
        Perform nonlinear time-history analysis on MDOF
    
        Parameters
        ----------
        fnames:                        list                List of the filepaths to the ground motions to be applied in the X Y and Z. At least the X direction is required.
        dt_gm:                        float                Time-step of the ground motions.
        sf:                           float                Scale factor to be applied to the records (Typically equal to 9.81).
        t_max:                        float                Duration of the record.
        dt_ansys:                     float                Time-step at which to conduct the analysis (Typically smaller than the record dt).
        nrha_outdir:                 string                Filepath where "TEMPORARY" files are saved and then deleted
        pflag:                         bool                Flag to print (or not) the nonlinear time-history analysis steps (Default is 'True').
        xi:                           float                The value of inherent damping (Default is 5% '0.05').
        ansys_soe:                   string                System of equations type. (Default is 'BandGeneral')
        constraints_handler:         string                The constraints handler object determines how the constraint equations are enforced in the analysis. Constraint equations enforce a specified value for a DOF, or a relationship between DOFs (Default is 'Plain').
        numberer:                    string                The DOF numberer object determines the mapping between equation numbers and degrees-of-freedom – how degrees-of-freedom are numbered (Default is 'RCM').
        test_type:                   string                This command is used to construct the LinearSOE and LinearSolver objects to store and solve the test of equations in the analysis (Default is 'NormDispIncr').
        init_tol:                     float                Tolerance criteria used to check for convergence (Default is 1e-6).
        init_iter:                    float                Max number of iterations to check (Default is 50).
        algorithm_type:              string                The integrator object determines the meaning of the terms in the system of equation object Ax=B (Default is 'Newton').
                        
        Returns
        -------
        control_nodes:                 list                List of MDOF system floor nodes
        conv_index:                    list                List containing whether or not analysis has converged (collapse index)
        node_disps:                   array                Nodal displacements
        node_accels:                  array                Nodal accelerations
        peak_drift:                   array                Peak storey drift values (i.e., all storeys per record)
        peak_accel:                   array                Peak floor acceleration values (i.e., all floors per record)
        max_peak_drift:               array                Maximum peak storey drift values (i.e., maximum of all storeys per record)
        max_peak_drift_dir:           array                Direction of maximum peak storey drift value (i.e., X or Y)
        max_peak_drift_loc:           array                Location of maximum peak storey drift values (i.e., storey ID)
        max_peak_accel:               array                Maximum peak floor acceleration values (i.e., maximum of all floors per record)
        max_peak_accel_dir:           array                Direction of maximum peak floor acceleration value (i.e., X or Y)
        max_peak_accel_loc:           array                Location of maximum peak floor acceleration values (i.e., floor ID) 
        peak_disp:                    array                Peak displacement values (i.e., all floors per record) 
    
        """        
    
        # define control nodes
        control_nodes = ops.getNodeTags()

        # Define the timeseries and patterns first
        if len(fnames) > 0:
            nrha_tsTagX = 1
            nrha_pTagX = 1
            ops.timeSeries('Path', nrha_tsTagX, '-dt', dt_gm, '-filePath', fnames[0], '-factor', sf) 
            ops.pattern('UniformExcitation', nrha_pTagX, 1, '-accel', nrha_tsTagX)
            ops.recorder('Node', '-file', f"{nrha_outdir}/floor_accel_X.txt", '-timeSeries', nrha_tsTagX, '-node', *control_nodes, '-dof', 1, 'accel')
        if len(fnames) > 1:
            nrha_tsTagY = 2
            nrha_pTagY = 2
            ops.timeSeries('Path', nrha_tsTagY, '-dt', dt_gm, '-filePath', fnames[1], '-factor', sf) 
            ops.pattern('UniformExcitation', nrha_pTagY, 2, '-accel', nrha_tsTagY)
            ops.recorder('Node', '-file', f"{nrha_outdir}/floor_accel_Y.txt", '-timeSeries', nrha_tsTagY, '-node', *control_nodes, '-dof', 2, 'accel')
        if len(fnames) > 2:
            nrha_tsTagZ = 3
            nrha_pTagZ = 3
            ops.timeSeries('Path', nrha_tsTagZ, '-dt', dt_gm, '-filePath', fnames[2], '-factor', sf) 
            ops.pattern('UniformExcitation', nrha_pTagZ, 3, '-accel', nrha_tsTagZ)
                
        # Set up the initial objects
        ops.system(ansys_soe)
        ops.constraints(constraints_handler)
        ops.numberer(numberer)
        ops.test(test_type, init_tol, init_iter)        
        ops.algorithm(algorithm_type)
        ops.integrator('Newmark', 0.5, 0.25)
        ops.analysis('Transient')
        
        # Set up analysis parameters
        conv_index = 0   # Initially define the collapse index (-1 for non-converged, 0 for stable)
        control_time = 0.0 
        ok = 0 # Set the convergence to 0 (initially converged)
        
        # Parse the data about the building
        top_nodes = control_nodes[1:]
        bottom_nodes = control_nodes[0:-1]
        h = []
        for i in np.arange(len(top_nodes)):
            topZ = ops.nodeCoord(top_nodes[i], 3)
            bottomZ = ops.nodeCoord(bottom_nodes[i], 3)
            dist = topZ - bottomZ
            if dist == 0:
                print("WARNING: Zero length found in drift check, using very large distance 1e9 instead")
                h.append(1e9)
            else:
                h.append(dist)
        
        # Create some arrays to record to
        peak_disp = np.zeros((len(control_nodes), 2))
        peak_drift = np.zeros((len(top_nodes), 2))
        peak_accel = np.zeros((len(top_nodes)+1, 2))
        
        # Set damping 
        if self.number_storeys == 1:
            
            #Set damping 
            alphaM = 2*self.omega[0]*xi
            ops.rayleigh(alphaM,0,0,0)
            
        else:
        
            alphaM = 2*self.omega[0]*self.omega[2]*xi/(self.omega[0] + self.omega[2])
            alphaK = 2*xi/(self.omega[0] + self.omega[2])   
            ops.rayleigh(alphaM,0,alphaK,0)
        
        
        
        # Define parameters for deformation animation
        # n_steps = int(np.ceil(t_max/dt_gm))+1
        # node_disps = np.zeros([n_steps,len(control_nodes)])
        # node_accels= np.zeros([n_steps,len(control_nodes)])
        
        # Run the actual analysis
        # step = 0 # initialise the step counter
        while conv_index == 0 and control_time <= t_max and ok == 0:
            ok = ops.analyze(1, dt_ansys)
            control_time = ops.getTime()
            
            if pflag is True:
                print('Completed {:.3f}'.format(control_time) + ' of {:.3f} seconds'.format(t_max) )
        
            # If the analysis fails, try the following changes to achieve convergence
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Trying reducing time-step in half...')
                ok = ops.analyze(1, 0.5*dt_ansys)
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Trying reducing time-step in quarter...')
                ok = ops.analyze(1, 0.25*dt_ansys)
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Trying relaxing convergence with more iterations...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ok = ops.analyze(1, 0.5*dt_ansys)
                ops.test(test_type, init_tol, init_iter)
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Trying relaxing convergence with more iteration and Newton with initial then current...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ops.algorithm('Newton', 'initialThenCurrent')
                ok = ops.analyze(1, 0.5*dt_ansys)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Trying relaxing convergence with more iteration and Newton with initial...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ops.algorithm('Newton', 'initial')
                ok = ops.analyze(1, 0.5*dt_ansys)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Attempting a Hail Mary...')
                ops.test('FixedNumIter', init_iter*10)
                ok = ops.analyze(1, 0.5*dt_ansys)
                ops.test(test_type, init_tol, init_iter)
                
            # Game over......
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Exiting analysis...')
                conv_index = -1
            
            # For each of the nodes to monitor, get the current drift
            for i in np.arange(len(top_nodes)):
                
                # Get the current storey drifts - absolute difference in displacement over the height between them
                curr_drift_X = np.abs(ops.nodeDisp(top_nodes[i], 1) - ops.nodeDisp(bottom_nodes[i], 1))/h[i]
                curr_drift_Y = np.abs(ops.nodeDisp(top_nodes[i], 2) - ops.nodeDisp(bottom_nodes[i], 2))/h[i]
        
                # Check if the current drift is greater than the previous peaks at the same storey
                if curr_drift_X > peak_drift[i, 0]:
                    peak_drift[i, 0] = curr_drift_X
                
                if curr_drift_Y > peak_drift[i, 1]:
                    peak_drift[i, 1] = curr_drift_Y
                    
            # For each node to monitor, get is absolute displacement
            for i in np.arange(len(control_nodes)):
                
                curr_disp_X = np.abs(ops.nodeDisp(control_nodes[i], 1))
                curr_disp_Y = np.abs(ops.nodeDisp(control_nodes[i], 2))
                
                # # Append the node displacements and accelerations (NOTE: Might change when bidirectional records are applied)
                # node_disps[step,i]  = ops.nodeDisp(control_nodes[i],1)
                # node_accels[step,i] = ops.nodeResponse(control_nodes[i],1, 3)
                
                # Check if the current drift is greater than the previous peaks at the same storey
                if curr_disp_X > peak_disp[i, 0]:
                    peak_disp[i, 0] = curr_disp_X
                
                if curr_disp_Y > peak_disp[i, 1]:
                    peak_disp[i, 1] = curr_disp_Y
                                                
        # Now that the analysis is finished, get the maximum in either direction and report the location also
        max_peak_drift = np.max(peak_drift)
        ind = np.where(peak_drift == max_peak_drift)
        if ind[1][0] == 0:
            max_peak_drift_dir = 'X'
        elif ind[1][0] == 1:
            max_peak_drift_dir = 'Y'
        max_peak_drift_loc = ind[0][0]+1
        
        # Get the floor accelerations. Need to use a recorder file because a direct query would return relative values
        ops.wipe() # First wipe to finish writing to the file
        
        if len(fnames) > 0:
            temp1 = np.transpose(np.max(np.abs(np.loadtxt(f"{nrha_outdir}/floor_accel_X.txt")), 0))
            peak_accel[:,0] = temp1
            os.remove(f"{nrha_outdir}/floor_accel_X.txt")
        
        elif len(fnames) > 1:
            
            temp1 = np.transpose(np.max(np.abs(np.loadtxt(f"{nrha_outdir}/floor_accel_X.txt")), 0))
            temp2 = np.transpose(np.max(np.abs(np.loadtxt(f"{nrha_outdir}/floor_accel_Y.txt")), 0))
            peak_accel = np.stack([temp1, temp2], axis=1)
            os.remove(f"{nrha_outdir}/floor_accel_X.txt")
            os.remove(f"{nrha_outdir}/floor_accel_Y.txt")
        
        # Get the maximum in either direction and report the location also
        max_peak_accel = np.max(peak_accel)
        ind = np.where(peak_accel == max_peak_accel)
        if ind[1][0] == 0:
            max_peak_accel_dir = 'X'
        elif ind[1][0] == 1:
            max_peak_accel_dir = 'Y'
        max_peak_accel_loc = ind[0][0]
        
        # Give some feedback on what happened  
        if conv_index == -1:
            print('------ ANALYSIS FAILED --------')
        elif conv_index == 0:
            print('~~~~~~~ ANALYSIS SUCCESSFUL ~~~~~~~~~')
            
        if pflag is True:
            print('Final state = {:d} (-1 for non-converged, 0 for stable)'.format(conv_index))
            print('Maximum peak storey drift {:.3f} radians at storey {:d} in the {:s} direction (Storeys = 1, 2, 3,...)'.format(max_peak_drift, max_peak_drift_loc, max_peak_drift_dir))
            print('Maximum peak floor acceleration {:.3f} g at floor {:d} in the {:s} direction (Floors = 0(G), 1, 2, 3,...)'.format(max_peak_accel, max_peak_accel_loc, max_peak_accel_dir))
                
        # Give the outputs
        return control_nodes, conv_index, peak_drift, peak_accel, max_peak_drift, max_peak_drift_dir, max_peak_drift_loc, max_peak_accel, max_peak_accel_dir, max_peak_accel_loc, peak_disp
