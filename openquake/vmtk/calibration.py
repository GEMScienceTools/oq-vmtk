import numpy as np
from scipy.linalg import eigh

def calibrate_model(nst, gamma, sdof_capacity, isFrame, isSOS):    
    """
    Function to calibrate MDOF storey force-deformation relationships
    based on SDOF-based capacity functions
    -----
    Input
    -----
    nst: float
        Number of storeys
    gamma: float
        SDOF-MDOF transformation factor
    sdofCapArray: array
        SDOF spectral displacements-accelerations
    isFrame: bool
        Flag for framed or braced building class (True or False)
    isSOS: bool
        Flag for building class or model containing a soft-storey or not (True or False)
    ------
    Output
    ------
    flm_mdof: list 
        MDOF floor masses
    stD_mdof: list
        MDOF storey displacements
    stF_mdof: list
        MDOF storey forces
    phi_mdof: list
        MDOF expected mode shape
        
    """     
        
        
    # If the building has a soft storey
    if isSOS:
    
        # Define the mass identity matrix (diagonal matrix that have 1). It assumes again that all masser are uniform
        I = np.identity(nst)
        
        if nst > 1:
        
            I[-1,-1] = 0.75    
        
        # Define the stiffnes tri-diagonal matrix, which considers the stiffness to be uniform accross all stories
        ## Note: this may need to be changed later given that it does not apply to soft storeys
        
        # Initialize a zero matrix of size nst x nst
        K = np.zeros((nst, nst))
        
        # Fill the diagonal with 2k for all floors except the first and last, which get k
        np.fill_diagonal(K, 2)
        
        # For the last floors, the diagonal element is k, not 2k
        K[-1, -1] = 1
        
        K[0,0] = 1.20
        
    
        # Fill the off-diagonal elements with -k (coupling between adjacent floors)
        for i in range(nst - 1):
            K[i, i + 1] = -1
            K[i + 1, i] = -1
        
        
        # Find mode shape based on the fact that stiffness and mass are uniform across floors
        # Solving the generalized eigenvalue problem
        # eigh solves K*x = lambda*M*x, it returns eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(K, I)
        
        # The first mode corresponds to the smallest (positive) eigenvalue
        idx_min = np.argmin(eigenvalues)
        
        # Corresponding mode shape (eigenvector)
        first_mode = eigenvectors[:, idx_min]
        
        # Normalize the mode shape (optional: to make sure it's unit norm)
        phi_mdof = first_mode / first_mode[-1]
            
        # Calculate the mass at each floor node knowing the mode shape, effective mass (1 unit ton) and transformation factor
        mass = np.dot(np.dot(np.transpose(phi_mdof),I),phi_mdof)/np.power(np.dot(np.dot(np.transpose(phi_mdof),I),np.ones(nst)),2)
                                
        # Assign the MDOF mass        
        flm_mdof = (np.diagonal(I)*mass).tolist()
                
    elif isFrame and nst <= 12:
        
        phi_mdof = np.zeros(nst)
        
        for i in range(nst):
            
            phi_mdof[i] = ((i+1)/nst)**0.6
        
        # Assign the MDOF mass
        
        I = np.identity(nst) 
        
        if nst > 1:
        
            I[-1,-1] = 0.75
            
        mass = np.dot(np.dot(np.transpose(phi_mdof),I),phi_mdof)/np.power(np.dot(np.dot(np.transpose(phi_mdof),I),np.ones(nst)),2)
                
        flm_mdof = (np.diagonal(I)*mass).tolist()
        
    else:                         
    
        # Define the mass identity matrix (diagonal matrix that have 1). It assumes again that all masser are uniform
        I = np.identity(nst)
        
        if nst > 1:
        
            I[-1,-1] = 0.75    
        
        # Define the stiffnes tri-diagonal matrix, which considers the stiffness to be uniform accross all stories
        ## Note: this may need to be changed later given that it does not apply to soft storeys
        
        # Initialize a zero matrix of size nst x nst
        K = np.zeros((nst, nst))
        
        # Fill the diagonal with 2k for all floors except the first and last, which get k
        np.fill_diagonal(K, 2)
        
        # For the last floors, the diagonal element is k, not 2k
        K[-1, -1] = 1
    
        # Fill the off-diagonal elements with -k (coupling between adjacent floors)
        for i in range(nst - 1):
            K[i, i + 1] = -1
            K[i + 1, i] = -1
        
        
        # Find mode shape based on the fact that stiffness and mass are uniform across floors
        # Solving the generalized eigenvalue problem
        # eigh solves K*x = lambda*M*x, it returns eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(K, I)
        
        # The first mode corresponds to the smallest (positive) eigenvalue
        idx_min = np.argmin(eigenvalues)
        
        # Corresponding mode shape (eigenvector)
        first_mode = eigenvectors[:, idx_min]
        
        # Normalize the mode shape (optional: to make sure it's unit norm)
        phi_mdof = first_mode / first_mode[-1]
    
        # Calculate the mass at each floor node knowing the mode shape, effective mass (1 unit ton) and transformation factor
        mass = np.dot(np.dot(np.transpose(phi_mdof),I),phi_mdof)/np.power(np.dot(np.dot(np.transpose(phi_mdof),I),np.ones(nst)),2)
                                
        # Assign the MDOF mass        
        flm_mdof = (np.diagonal(I)*mass).tolist()
                
    if nst == 1:
        
        gamma = 1.0   

    ### Get the MDOF Capacity Curves Storey-Deformation Relationship
    rows, columns = np.shape(sdof_capacity)
    stD_mdof = np.zeros([nst,rows])
    stF_mdof = np.zeros([nst,rows])
    
    if len(sdof_capacity) == 3: # In case of trilinear capacity curve
        
        for i in range(nst):
            
            if i == 0:
                
                # get the force or spectral acceleration arrays at each storey
                # Note here that since we assume a mode shape and masses, then the real gamma
                # is different from the one in csv files. Technically, the multiplication outcome
                # of the factors after sdofCapArray should be equal to 1.0, which is the effective
                # mass of SDoF system
                stF_mdof[i,:] = sdof_capacity[:,1]*gamma*np.dot(np.dot(np.transpose(phi_mdof),I*mass),np.ones(nst))
                
                # get the displacement or spectral displacement arrays at each storey
                stD_mdof[i,:] = sdof_capacity[:,0]*gamma*phi_mdof[i]
                
                # # Fix the initial stiffness to get the same period as the SDoF system
                # stD_mdof[i,0] = stF_mdof[i,0]/(k0/9.81)
                # ## NOTE: the k0 is divided by 9.81 to maintain unit consistency because
                # ## Moe multiplies all y-axis by g later in opensees
                
                # # Find the slope of the second branch of the capacity curve of first floor
                # # to use it for predicting displacements of the other floors
                # slope_2nd = (stF_mdof[0,1] - stF_mdof[0,0])/(stD_mdof[0,1] - stD_mdof[0,0])
                
                
            
            else:
                
                # Find the force contribution ratio, based on the mode shape (it works as
                # long as the masses are uniform across the floors).
                
                # This Ratio is disabled for now as we want the springs to have the full srength. The amount
                # of shear developing there will be determined by the analysis and force distribution. So it
                # is wrong to use the below ratio. Even Lu et al (2014) has removed this ratio in his later papers
                Ratio_force = np.sum(phi_mdof[i:]*flm_mdof[i:])/np.sum(phi_mdof*flm_mdof)
                
                # Multiply the Ratio with the Y-coordinates of the first floor (as it has the full base shear)
                stF_mdof[i,:] = stF_mdof[0,:]*Ratio_force 
                
                
                # Find the interstorey drift contribution ratio to be multiplied by the
                # interstorey drift of the first floor to give us the interstorey drift for
                # the remaining floors. If the mode shape is linear, this ratio will be
                # always one, so the storeys are drifting the same way. Note that floor
                # height is always assumed constant
                Ratio_disp = (phi_mdof[i]-phi_mdof[i-1])/phi_mdof[0]
                
                
                # Derive the displacements 
                stD_mdof[i,:] = stD_mdof[0,:]*Ratio_disp
                
                # # Fix the initial stiffness to be the same as the first floor
                # stD_mdof[i,0] = stF_mdof[i,0]/(k0/9.81)
    
                # # Find the displacement of the second branch
                # stD_mdof[i,1] = stD_mdof[i,0] + (stF_mdof[i,1] - stF_mdof[i,0])/slope_2nd
                
                # The last displacement will stay the same
                
                # This is for the case that the ultimate displacement is less than
                # the previous ones. I basically scale the previous displacements
                # with the same scale of the ultimate displacement rather than
                # computing the previous displacements by maintaining the same
                # slopes as the first floor
                if stD_mdof[i,2] < stD_mdof[i,1]:
                    
                    stD_mdof[i,:] = stD_mdof[0,:]*Ratio_disp
                
                
    elif len(sdof_capacity) == 2: # In case of bilinear capacity curve
                
                
        for i in range(nst):
            
            if i == 0:
                
                # get the force or spectral acceleration arrays at each storey
                # Note here that since we assume a mode shape and masses, then the real gamma
                # is different from the one in csv files. Technically, the multiplication outcome
                # of the factors after sdofCapArray should be equal to 1.0, which is the effective
                # mass of SDoF system
                stF_mdof[i,:] = sdof_capacity[:,1]*gamma*np.dot(np.dot(np.transpose(phi_mdof),I*mass),np.ones(nst))
                
                
                # get the displacement or spectral displacement arrays at each storey
                stD_mdof[i,:] = sdof_capacity[:,0]*gamma*phi_mdof[i]
                
                # # Fix the initial stiffness to get the same period as the SDoF system
                # stD_mdof[i,0] = stF_mdof[i,0]/(k0/9.81)
                # ## NOTE: the k0 is divided by 9.81 to maintain unit consistency because
                # ## Moe multiplies all y-axis by g later in opensees
                
            else:
                
                # Find the force contribution ratio, based on the mode shape (it works as
                # long as the masses are uniform across the floors).
                
                # This Ratio is disabled for now as we want the springs to have the full srength. The amount
                # of shear developing there will be determined by the analysis and force distribution. So it
                # is wrong to use the below ratio. Even Lu et al (2014) has removed this ratio in his later papers
                Ratio_force = np.sum(phi_mdof[i:]*flm_mdof[i:])/np.sum(phi_mdof*flm_mdof)
                
                # Multiply the Ratio with the Y-coordinates of the first floor (as it has the full base shear)
                stF_mdof[i,:] = stF_mdof[0,:]*Ratio_force 
                                
                # Find the interstorey drift contribution ratio to be multiplied by the
                # interstorey drift of the first floor to give us the interstorey drift for
                # the remaining floors. If the mode shape is linear, this ratio will be
                # always one, so the storeys are drifting the same way. Note that floor
                # height is always assumed constant
                Ratio_disp = (phi_mdof[i]-phi_mdof[i-1])/phi_mdof[0]
                                
                # Derive the displacements 
                stD_mdof[i,:] = stD_mdof[0,:]*Ratio_disp
                
                # # Fix the initial stiffness to be the same as the first floor
                # stD_mdof[i,0] = stF_mdof[i,0]/(k0/9.81)
                
                # This is for the case that the ultimate displacement is less than
                # the previous ones. I basically scale the previous displacements
                # with the same scale of the ultimate displacement rather than
                # computing the previous displacements by maintaining the same
                # slopes as the first floor
                if stD_mdof[i,1] < stD_mdof[i,0]:
                    
                    stD_mdof[i,:] = stD_mdof[0,:]*Ratio_disp
   
 
    if len(sdof_capacity) == 4: # In case of quadrilinear capacity curve
        
        for i in range(nst):
            
            if i == 0:
                
                # get the force or spectral acceleration arrays at each storey
                # Note here that since we assume a mode shape and masses, then the real gamma
                # is different from the one in csv files. Technically, the multiplication outcome
                # of the factors after sdofCapArray should be equal to 1.0, which is the effective
                # mass of SDoF system
                stF_mdof[i,:] = sdof_capacity[:,1]*gamma*np.dot(np.dot(np.transpose(phi_mdof),I*mass),np.ones(nst))
                
                
                # get the displacement or spectral displacement arrays at each storey
                stD_mdof[i,:] = sdof_capacity[:,0]*gamma*phi_mdof[i]
                                
        
            else:
                
                # Find the force contribution ratio, based on the mode shape (it works as
                # long as the masses are uniform across the floors).
                
                # This Ratio is disabled for now as we want the springs to have the full srength. The amount
                # of shear developing there will be determined by the analysis and force distribution. So it
                # is wrong to use the below ratio. Even Lu et al (2014) has removed this ratio in his later papers
                Ratio_force = np.sum(phi_mdof[i:]*flm_mdof[i:])/np.sum(phi_mdof*flm_mdof)
                
                
                # Multiply the Ratio with the Y-coordinates of the first floor (as it has the full base shear)
                stF_mdof[i,:] = stF_mdof[0,:]*Ratio_force 
                
                
                # Find the interstorey drift contribution ratio to be multiplied by the
                # interstorey drift of the first floor to give us the interstorey drift for
                # the remaining floors. If the mode shape is linear, this ratio will be
                # always one, so the storeys are drifting the same way. Note that floor
                # height is always assumed constant
                Ratio_disp = (phi_mdof[i]-phi_mdof[i-1])/phi_mdof[0]
                
                
                # Derive the displacements 
                stD_mdof[i,:] = stD_mdof[0,:]*Ratio_disp
                
                # This is for the case that the ultimate displacement is less than
                # the previous ones. I basically scale the previous displacements
                # with the same scale of the ultimate displacement rather than
                # computing the previous displacements by maintaining the same
                # slopes as the first floor
                if stD_mdof[i,3] < stD_mdof[i,2]:
                    
                    stD_mdof[i,:] = stD_mdof[0,:]*Ratio_disp
                          
        
    return flm_mdof, stD_mdof, stF_mdof, phi_mdof