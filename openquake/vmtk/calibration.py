import numpy as np
from scipy.linalg import eigh


def calibrate_model(number_storeys, gamma, sdof_capacity, sdof_period, isFrame, isSOS):    

    """
    Function to calibrate MDOF storey force-deformation relationships
    based on SDOF-based capacity functions
    ----------
    Parameters
    ----------
    number_storeys             int         Number of storeys
    gamma:                   float         First-mode based transformation factor 
    sdof_capacity:           array         SDOF spectral displacements-accelerations
    sdof_period              float         SDOF fundamental period
    isFrame                   bool         Flag for building class or model containing moment resisting frames (True or False)
    isSOS                     bool         Flag for building class or model containing soft-storey (True or False)
    -------
    Returns
    -------
    floor_masses:             list         Resulting floor masses in tonnes
    storey_disps:             list         Storey displacements
    storey_forces             list         Storey forces
    mdof_phi                  list         MDOF mode shape        
    """     
        
        
    # If the building has a soft storey
    if isSOS:
    
        # Define the mass identity matrix (diagonal matrix that have 1). It assumes again that all masser are uniform
        I = np.identity(number_storeys)
        
        if number_storeys > 1:
        
            I[-1,-1] = 0.75    
        
        # Define the stiffnes tri-diagonal matrix, which considers the stiffness to be uniform accross all stories
        ## Note: this may need to be changed later given that it does not apply to soft storeys
        
        # Initialize a zero matrix of size number_storeys x number_storeys
        K = np.zeros((number_storeys, number_storeys))
        
        # Fill the diagonal with 2k for all floors except the first and last, which get k
        np.fill_diagonal(K, 2)
        
        # For the last floors, the diagonal element is k, not 2k
        K[-1, -1] = 1
        
        K[0,0] = 1.20
        
    
        # Fill the off-diagonal elements with -k (coupling between adjacent floors)
        for i in range(number_storeys - 1):
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
        mdof_phi = first_mode / first_mode[-1]
        
        # Calculate the sum of the squares of phi
        sum_square_phi = np.dot(mdof_phi, mdof_phi)
    
        # Calculate the sum of phi and then square it
        sum_phi_square = np.power(np.sum(mdof_phi), 2)
        
        # Calculate the sum of mode shape
        sum_phi = np.sum(mdof_phi)
        
        # Calculate the mass at each floor node knowing the mode shape, effective mass (1 unit ton) and transformation factor
        mass = np.dot(np.dot(np.transpose(mdof_phi),I),mdof_phi)/np.power(np.dot(np.dot(np.transpose(mdof_phi),I),np.ones(number_storeys)),2)
        
        # mass = 1/(sum_square_phi*gamma)
        
        # Real Value of Gamma because of the asssumed mode shape
        gamma_real = np.dot(np.dot(np.transpose(mdof_phi),I),np.ones(number_storeys))/np.dot(np.dot(np.transpose(mdof_phi),I),mdof_phi)
                
        # Assign the MDOF mass
        
        floor_masses = (np.diagonal(I)*mass).tolist()
        # floor_masses = [mass]*number_storeys
        
        # Compute Lambda as per Lu et al (pay attention as one of the papers has a mistake)
        lamda = np.dot(np.dot(np.transpose(mdof_phi),I),mdof_phi)/np.dot(np.dot(np.transpose(mdof_phi),K),mdof_phi)
        
    elif isFrame and number_storeys <= 12:
        
        mdof_phi = np.zeros(number_storeys)
        
        for i in range(number_storeys):
            
            mdof_phi[i] = ((i+1)/number_storeys)**0.6
        
        # Assign the MDOF mass
        
        I = np.identity(number_storeys) 
        
        if number_storeys > 1:
        
            I[-1,-1] = 0.75
            
        
        #     # floor_masses = [mass]*number_storeys
        
        mass = np.dot(np.dot(np.transpose(mdof_phi),I),mdof_phi)/np.power(np.dot(np.dot(np.transpose(mdof_phi),I),np.ones(number_storeys)),2)
        
        # floor_masses = [mass]*number_storeys
        
        floor_masses = (np.diagonal(I)*mass).tolist()
        
        gamma_real = np.dot(np.dot(np.transpose(mdof_phi),I),np.ones(number_storeys))/np.dot(np.dot(np.transpose(mdof_phi),I),mdof_phi)
              


    else:                         
    
        # Define the mass identity matrix (diagonal matrix that have 1). It assumes again that all masser are uniform
        I = np.identity(number_storeys)
        
        if number_storeys > 1:
        
            I[-1,-1] = 0.75    
        
        # Define the stiffnes tri-diagonal matrix, which considers the stiffness to be uniform accross all stories
        ## Note: this may need to be changed later given that it does not apply to soft storeys
        
        # Initialize a zero matrix of size number_storeys x number_storeys
        K = np.zeros((number_storeys, number_storeys))
        
        # Fill the diagonal with 2k for all floors except the first and last, which get k
        np.fill_diagonal(K, 2)
        
        # For the last floors, the diagonal element is k, not 2k
        K[-1, -1] = 1
    
        # Fill the off-diagonal elements with -k (coupling between adjacent floors)
        for i in range(number_storeys - 1):
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
        mdof_phi = first_mode / first_mode[-1]
    
    
        # Calculate the sum of the squares of phi
        sum_square_phi = np.dot(mdof_phi, mdof_phi)
    
        # Calculate the sum of phi and then square it
        sum_phi_square = np.power(np.sum(mdof_phi), 2)
        
        # Calculate the sum of mode shape
        sum_phi = np.sum(mdof_phi)
        
        # Calculate the mass at each floor node knowing the mode shape, effective mass (1 unit ton) and transformation factor
        mass = np.dot(np.dot(np.transpose(mdof_phi),I),mdof_phi)/np.power(np.dot(np.dot(np.transpose(mdof_phi),I),np.ones(number_storeys)),2)
        
        # mass = 1/(sum_square_phi*gamma)
        
        # Real Value of Gamma because of the asssumed mode shape
        gamma_real = np.dot(np.dot(np.transpose(mdof_phi),I),np.ones(number_storeys))/np.dot(np.dot(np.transpose(mdof_phi),I),mdof_phi)
                
        # Assign the MDOF mass
        
        floor_masses = (np.diagonal(I)*mass).tolist()

        
    if number_storeys == 1:
        
        gamma = 1.0   

    ### Get the MDOF Capacity Curves Storey-Deformation Relationship
    rows, columns = np.shape(sdof_capacity)
    storey_disps = np.zeros([number_storeys,rows])
    storey_forces = np.zeros([number_storeys,rows])



    # Compute the interstorey initial stiffness (in kN/m)
    # k0 = lamda*4*pi**2*mass/sdof_period**2
    
    if len(sdof_capacity) == 3: # In case of trilinear capacity curve
        
        for i in range(number_storeys):
            
            if i == 0:
                
                # get the force or spectral acceleration arrays at each storey
                # Note here that since we assume a mode shape and masses, then the real gamma
                # is different from the one in csv files. Technically, the multiplication outcome
                # of the factors after sdof_capacity should be equal to 1.0, which is the effective
                # mass of SDoF system
                storey_forces[i,:] = sdof_capacity[:,1]*gamma*np.dot(np.dot(np.transpose(mdof_phi),I*mass),np.ones(number_storeys))
                
                # get the displacement or spectral displacement arrays at each storey
                storey_disps[i,:] = sdof_capacity[:,0]*gamma*mdof_phi[i]
                              
                
            
            else:
                
                # Find the force contribution ratio, based on the mode shape (it works as
                # long as the masses are uniform across the floors).
                
                # This Ratio is disabled for now as we want the springs to have the full srength. The amount
                # of shear developing there will be determined by the analysis and force distribution. So it
                # is wrong to use the below ratio. Even Lu et al (2014) has removed this ratio in his later papers
                Ratio_force = np.sum(mdof_phi[i:]*floor_masses[i:])/np.sum(mdof_phi*floor_masses)
                
                # Multiply the Ratio with the Y-coordinates of the first floor (as it has the full base shear)
                storey_forces[i,:] = storey_forces[0,:]*Ratio_force 
                
                
                # Find the interstorey drift contribution ratio to be multiplied by the
                # interstorey drift of the first floor to give us the interstorey drift for
                # the remaining floors. If the mode shape is linear, this ratio will be
                # always one, so the storeys are drifting the same way. Note that floor
                # height is always assumed conumber_storeysant
                Ratio_disp = (mdof_phi[i]-mdof_phi[i-1])/mdof_phi[0]
                
                
                # Derive the displacements 
                storey_disps[i,:] = storey_disps[0,:]*Ratio_disp
                

                
                # This is for the case that the ultimate displacement is less than
                # the previous ones. I basically scale the previous displacements
                # with the same scale of the ultimate displacement rather than
                # computing the previous displacements by maintaining the same
                # slopes as the first floor
                if storey_disps[i,2] < storey_disps[i,1]:
                    
                    storey_disps[i,:] = storey_disps[0,:]*Ratio_disp
                
                
    elif len(sdof_capacity) == 2: # In case of bilinear capacity curve
                
                
        for i in range(number_storeys):
            
            if i == 0:
                
                # get the force or spectral acceleration arrays at each storey
                # Note here that since we assume a mode shape and masses, then the real gamma
                # is different from the one in csv files. Technically, the multiplication outcome
                # of the factors after sdof_capacity should be equal to 1.0, which is the effective
                # mass of SDoF system
                storey_forces[i,:] = sdof_capacity[:,1]*gamma*np.dot(np.dot(np.transpose(mdof_phi),I*mass),np.ones(number_storeys))
                
                
                # get the displacement or spectral displacement arrays at each storey
                storey_disps[i,:] = sdof_capacity[:,0]*gamma*mdof_phi[i]
                
                # # Fix the initial stiffness to get the same period as the SDoF system
                # storey_disps[i,0] = storey_forces[i,0]/(k0/9.81)
                # ## NOTE: the k0 is divided by 9.81 to maintain unit consistency because
                # ## Moe multiplies all y-axis by g later in opensees
                
            else:
                
                # Find the force contribution ratio, based on the mode shape (it works as
                # long as the masses are uniform across the floors).
                
                # This Ratio is disabled for now as we want the springs to have the full srength. The amount
                # of shear developing there will be determined by the analysis and force distribution. So it
                # is wrong to use the below ratio. Even Lu et al (2014) has removed this ratio in his later papers
                Ratio_force = np.sum(mdof_phi[i:]*floor_masses[i:])/np.sum(mdof_phi*floor_masses)
                
                # Multiply the Ratio with the Y-coordinates of the first floor (as it has the full base shear)
                storey_forces[i,:] = storey_forces[0,:]*Ratio_force 
                                
                # Find the interstorey drift contribution ratio to be multiplied by the
                # interstorey drift of the first floor to give us the interstorey drift for
                # the remaining floors. If the mode shape is linear, this ratio will be
                # always one, so the storeys are drifting the same way. Note that floor
                # height is always assumed conumber_storeysant
                Ratio_disp = (mdof_phi[i]-mdof_phi[i-1])/mdof_phi[0]
                                
                # Derive the displacements 
                storey_disps[i,:] = storey_disps[0,:]*Ratio_disp
                
                
                # This is for the case that the ultimate displacement is less than
                # the previous ones. I basically scale the previous displacements
                # with the same scale of the ultimate displacement rather than
                # computing the previous displacements by maintaining the same
                # slopes as the first floor
                if storey_disps[i,1] < storey_disps[i,0]:
                    
                    storey_disps[i,:] = storey_disps[0,:]*Ratio_disp
   
 
    if len(sdof_capacity) == 4: # In case of quadrilinear capacity curve
        
        for i in range(number_storeys):
            
            if i == 0:
                
                # get the force or spectral acceleration arrays at each storey
                # Note here that since we assume a mode shape and masses, then the real gamma
                # is different from the one in csv files. Technically, the multiplication outcome
                # of the factors after sdof_capacity should be equal to 1.0, which is the effective
                # mass of SDoF system
                storey_forces[i,:] = sdof_capacity[:,1]*gamma*np.dot(np.dot(np.transpose(mdof_phi),I*mass),np.ones(number_storeys))
                
                
                # get the displacement or spectral displacement arrays at each storey
                storey_disps[i,:] = sdof_capacity[:,0]*gamma*mdof_phi[i]
                               
            
            else:
                
                # Find the force contribution ratio, based on the mode shape (it works as
                # long as the masses are uniform across the floors).
                
                # This Ratio is disabled for now as we want the springs to have the full srength. The amount
                # of shear developing there will be determined by the analysis and force distribution. So it
                # is wrong to use the below ratio. Even Lu et al (2014) has removed this ratio in his later papers
                Ratio_force = np.sum(mdof_phi[i:]*floor_masses[i:])/np.sum(mdof_phi*floor_masses)
                
                # Multiply the Ratio with the Y-coordinates of the first floor (as it has the full base shear)
                storey_forces[i,:] = storey_forces[0,:]*Ratio_force 
                
                
                # Find the interstorey drift contribution ratio to be multiplied by the
                # interstorey drift of the first floor to give us the interstorey drift for
                # the remaining floors. If the mode shape is linear, this ratio will be
                # always one, so the storeys are drifting the same way. Note that floor
                # height is always assumed conumber_storeysant
                Ratio_disp = (mdof_phi[i]-mdof_phi[i-1])/mdof_phi[0]
                
                
                # Derive the displacements 
                storey_disps[i,:] = storey_disps[0,:]*Ratio_disp
                

                # This is for the case that the ultimate displacement is less than
                # the previous ones. I basically scale the previous displacements
                # with the same scale of the ultimate displacement rather than
                # computing the previous displacements by maintaining the same
                # slopes as the first floor
                if storey_disps[i,3] < storey_disps[i,2]:
                    
                    storey_disps[i,:] = storey_disps[0,:]*Ratio_disp
                          
        
    return floor_masses, storey_disps, storey_forces, mdof_phi
