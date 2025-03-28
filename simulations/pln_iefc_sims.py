import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import time
import sys
from scipy.linalg import hadamard
import poppy
from hcipy import *

Nact = 12
Nports = 4
context = None
BMC_to_waves = (3480e-9)/1568e-9; #3480 nm/BMC unit divided by working wavelength of 1568 nm
x_center = 0.06522580693476698
y_center = -0.13528315412186384

num_modes = 6
modes_shaped = np.load('shaped_fields.npy')
modes_raveled = np.reshape(modes_shaped,[modes_shaped.shape[0],modes_shaped.shape[1]**2]).T

#simulation setup

grid_size = 200
pupil_grid = make_pupil_grid(grid_size)

focal_dim = modes_shaped.shape[1]
focal_extent = 4.4
focal_grid_lant2 = make_uniform_grid([focal_dim,focal_dim],[focal_extent,focal_extent])
prop_lant2 = FraunhoferPropagator(pupil_grid, focal_grid_lant2)
dx_lant2 = focal_grid_lant2.x[1]-focal_grid_lant2.x[0]
dxi_lant2 = pupil_grid.x[1]-pupil_grid.x[0]

circ = make_circular_aperture(diameter=1)
aperture = circ(pupil_grid)

num_actuators = Nact**2
actuator_spacing = 1.05 / Nact
influence_functions = make_xinetics_influence_functions(pupil_grid, Nact, actuator_spacing)
dm1_optic = DeformableMirror(influence_functions)

#distance_between_dms = (200e-3)/(2*(7e-3)**2/700e-9) #nondimensionalized from hcipy tutorial
distance_between_dms = 0.5
prop_between_dms = FresnelPropagator(pupil_grid, distance_between_dms)
dm2_optic = DeformableMirror(influence_functions)

#lantern
lant2_modes = ModeBasis(modes_raveled,focal_grid_lant2)
lant2 = PhotonicLantern(lant2_modes)
lant2_thpts = np.array([0.8861, 0.8899, 0.7778, 0.5828, 0.6138, 0.6167])
lp_mode_names = ['LP 01', 'LP 11a', 'LP 11b', 'LP 21a', 'LP 21b', 'LP 02']

#perfect lantern
pln = PhotonicLanternNuller(pupil_grid,focal_grid_lant2)
lp_modes = make_lp_modes(focal_grid_lant2, 1.5 * np.pi, 1.31)

lp_modes_raveled = []
for n in range(len(lp_modes)):
    lp_modes_raveled.append(lp_modes[n].ravel())
lp_modes_raveled = np.array(lp_modes_raveled).T

#global wfe?
ptv = 0.1
wfe_global = None
#wfe_global = make_power_law_error(pupil_grid, ptv, 1)+1j*make_power_law_error(pupil_grid, ptv, 1)
#wfe_global = make_power_law_error(pupil_grid, ptv, 1)*2

#coronagraph
focal_grid_coro = make_focal_grid(4, 12)
prop_coro = FraunhoferPropagator(pupil_grid, focal_grid_coro)
coro = VortexCoronagraph(pupil_grid, 2)
lyot_mask = evaluate_supersampled(circular_aperture(0.95), pupil_grid, 4)
lyot_stop = Apodizer(lyot_mask)

#extended source params
extended_source = False
source_r = 0.005
phis = np.arange(0,2*np.pi,2*np.pi/6)
xs = x_center+source_r*np.cos(phis)
ys = y_center+source_r*np.sin(phis)
xs = np.append(xs,x_center)
ys = np.append(ys,y_center)

class SimDM():
    def __init__(self):
        self.current_surf = np.zeros((Nact,Nact))

    def getCurrentSurf(self):
        return self.current_surf

    def setSurf(self, array):
        self.current_surf = array.copy()
        return self.current_surf

    def zeroAll(self):
        self.current_surf = np.zeros((Nact,Nact))
        return self.current_surf

#hadamard mode generation adapted from lina
def create_hadamard_modes(): 
    Nacts = Nact**2
    np2 = 2**int(np.ceil(np.log2(Nacts)))
    hmodes = hadamard(np2)
    had_modes = hmodes[:,:Nacts]
    return had_modes
    
#Zernikes adapted from Dan's DM code, using RMS norm and not PV norm
def create_zernike_modes(min_noll,max_noll):
        
    # Generate the default Zernikes from poppy
    zern_basis = poppy.zernike.zernike_basis(max_noll, Nact+2,outside=0.0)
        
    # Remove unnecessary edges of 0 that poppy adds
    zern_basis = zern_basis[:,1:-1,1:-1]
        
    ## Normalize so that each zernike has RMS=1. Could probably do this vectorized
    #  # but for now I'll do it the lazy, for-loop way
    for ind in range(zern_basis.shape[0]):
        zern       = zern_basis[ind,:,:]
        zern_basis[ind,:,:] /= np.sqrt(np.mean(np.square(zern)))

    zern_basis = zern_basis[min_noll-1:]
    return zern_basis

def prop_pln_lant2(wfe = None, tt_x = 0, tt_y = 0, dm1_acts = None, dm2_acts = None, return_integrand = False):
    if wfe_global is not None:
        wf = Wavefront(aperture*np.exp(1j*wfe_global)*np.exp(1j*2*np.pi*(tt_x)*pupil_grid.x)*np.exp(1j*2*np.pi*(tt_y)*pupil_grid.y))
    else:
        wf = Wavefront(aperture*np.exp(1j*2*np.pi*(tt_x)*pupil_grid.x)*np.exp(1j*2*np.pi*(tt_y)*pupil_grid.y))

    #if DM 1
    if dm1_acts is not None:
        dm1_optic.actuators = dm1_acts
        if dm2_acts is not None:
            dm2_optic.actuators = dm2_acts
            wf = prop_between_dms.backward(dm2_optic(prop_between_dms.forward(dm1_optic(wf))))
        else: 
            wf = dm1_optic(wf)

    foc = prop_lant2.forward(wf)
    output = lant2.forward(foc)

    integrand = np.expand_dims(foc.electric_field.conj(),0)*modes_raveled.T

    if return_integrand:
        return output, integrand
    else:
        return output

def prop_pln_lant2_extended(wfe = None, tt_x = 0, tt_y = 0, dm1_acts = None, dm2_acts = None, return_integrand = False):
    output = []
    for n in range(len(xs)):
        tt_x = xs[n]
        tt_y = ys[n]
        if wfe_global is not None:
            wf = Wavefront(aperture*np.exp(1j*wfe_global)*np.exp(1j*2*np.pi*(tt_x)*pupil_grid.x)*np.exp(1j*2*np.pi*(tt_y)*pupil_grid.y))
        else:
            wf = Wavefront(aperture*np.exp(1j*2*np.pi*(tt_x)*pupil_grid.x)*np.exp(1j*2*np.pi*(tt_y)*pupil_grid.y))

        wf.total_intensity = 1/7

        #if DM 1
        if dm1_acts is not None:
            dm1_optic.actuators = dm1_acts
            if dm2_acts is not None:
                dm2_optic.actuators = dm2_acts
                wf = prop_between_dms.backward(dm2_optic(prop_between_dms.forward(dm1_optic(wf))))
            else: 
                wf = dm1_optic(wf)

        foc = prop_lant2.forward(wf)
        output.append(lant2.forward(foc).intensity)
    output = np.array(output)
    output = np.sum(output,axis=0)

    return output

def prop_pln_perfect(wfe = None, tt_x = 0, tt_y = 0, dm1_acts = None, dm2_acts = None, return_integrand = False):
    if wfe_global is not None:
        wf = Wavefront(aperture*np.exp(1j*wfe_global)*np.exp(1j*2*np.pi*(tt_x)*pupil_grid.x)*np.exp(1j*2*np.pi*(tt_y)*pupil_grid.y))
    else:
        wf = Wavefront(aperture*np.exp(1j*2*np.pi*(tt_x)*pupil_grid.x)*np.exp(1j*2*np.pi*(tt_y)*pupil_grid.y))

    #if DM 1
    if dm1_acts is not None:
        dm1_optic.actuators = dm1_acts
        if dm2_acts is not None:
            dm2_optic.actuators = dm2_acts
            wf = prop_between_dms.backward(dm2_optic(prop_between_dms.forward(dm1_optic(wf))))
        else: 
            wf = dm1_optic(wf)

    output = pln.forward(wf)

    foc = prop_lant2.forward(wf)
    integrand = np.expand_dims(foc.electric_field.conj(),0)*lp_modes_raveled.T

    if return_integrand:
       return output, integrand
    else:
       return output
    return output

def prop_coronagraph(wfe = None, tt_x = 0, tt_y = 0, dm1_acts = None, dm2_acts = None):
    if wfe is not None:
        wf = Wavefront(aperture*np.exp(1j*wfe)*np.exp(1j*2*np.pi*(tt_x)*pupil_grid.x)*np.exp(1j*2*np.pi*(tt_y)*pupil_grid.y))
    else:
        wf = Wavefront(aperture*np.exp(1j*2*np.pi*(tt_x)*pupil_grid.x)*np.exp(1j*2*np.pi*(tt_y)*pupil_grid.y))

    #if DM 1
    if dm1_acts is not None:
        dm1_optic.actuators = dm1_acts
        if dm2_acts is not None:
            dm2_optic.actuators = dm2_acts
            wf = prop_between_dms.backward(dm2_optic(prop_between_dms.forward(dm1_optic(wf))))
        else: 
            wf = dm1_optic(wf)

    ls = lyot_stop.forward(coro.forward(wf))
    output = prop_coro.forward(ls)

    #integrand = np.expand_dims(foc.electric_field.conj(),0)*modes_raveled.T

    #if return_integrand:
    #    return output, integrand
    #else:
    #    return output
    return output

def get_photometry(context,navg,divisor=1e6,sleep_t=1e-12,wfe=None,use_photutils=False,perfect_lantern=False,extended_source=False):
    currentDM = dm.getCurrentSurf()
    if perfect_lantern is True:
        outputs = prop_pln_perfect(dm1_acts=currentDM.ravel(),wfe=wfe)
        outputs = outputs.intensity
        return outputs[1:5]
    else:
        if extended_source is True:
            outputs = prop_pln_lant2_extended(dm1_acts=currentDM.ravel(),wfe=wfe)
            outputs_lossy = outputs*lant2_thpts
            return outputs_lossy[1:5]
        else:
            outputs = prop_pln_lant2(tt_x=x_center,tt_y=y_center,dm1_acts=currentDM.ravel(),wfe=wfe)
            outputs = outputs.intensity
            outputs_lossy = outputs*lant2_thpts
            return outputs_lossy[1:5]

def get_photometry_coronagraph(context,navg,divisor=1e6,sleep_t=1e-12,wfe=None,use_photutils=False):
    currentDM = dm.getCurrentSurf()
    outputs = prop_coronagraph(tt_x=x_center,tt_y=y_center,dm1_acts=currentDM.ravel(),wfe=wfe)
    outputs = outputs.intensity
    
    return outputs

def get_photometry_2dm(context,navg,divisor=1e6,sleep_t=1e-12,use_photutils=False):
    currentDM1 = dm1.getCurrentSurf()
    currentDM2 = dm2.getCurrentSurf()
    outputs = prop_pln_lant2(tt_x=x_center,tt_y=y_center,dm1_acts=currentDM1.ravel(),dm2_acts=currentDM2.ravel())
    outputs = outputs.intensity
    outputs_lossy = outputs*lant2_thpts
    
    return outputs_lossy[1:5]
    
#iEFC code adapted from lina
def take_measurement(dm,probe_cube,probe_amplitude,navg=1,sleep_t=1e-12,control_ports=[0,1,2,3],wfe=None,return_all=False,perfect_lantern=False,extended_source=False,coronagraph=False):

    differential_operator = []
    for i in range(len(probe_cube)):
        vec = [0]*2*len(probe_cube)
        vec[2*i] = -1
        vec[2*i+1] = 1
        differential_operator.append(vec)
    differential_operator = np.array(differential_operator) / (2 * probe_amplitude)
    
    amps = np.linspace(-probe_amplitude, probe_amplitude, 2)
    images = []
    for probe in probe_cube: 
        for amp in amps:
            surf = dm.getCurrentSurf()
            dm.setSurf(surf+amp*probe)
            if coronagraph is True:
                image = get_photometry_coronagraph(context,navg,sleep_t=sleep_t,wfe=wfe)
            else:
                image = get_photometry(context,navg,sleep_t=sleep_t,wfe=wfe,perfect_lantern=perfect_lantern, extended_source=extended_source)
            image = image[control_ports]
            images.append(image)
            dm.setSurf(surf)
            #time.sleep(0.1)
    images = np.array(images)
    
    differential_images = differential_operator.dot(images)
    if return_all:
        return differential_images,images
    else:
        return differential_images

def take_measurement_2dm(dm1, dm2, probe_cube, probe_amplitude,navg=1,sleep_t=1e-12,control_ports=[0,1,2,3],return_all=False):

    differential_operator = []
    for i in range(len(probe_cube)):
        vec = [0]*2*len(probe_cube)
        vec[2*i] = -1
        vec[2*i+1] = 1
        differential_operator.append(vec)
    differential_operator = np.array(differential_operator) / (2 * probe_amplitude)

    amps = np.linspace(-probe_amplitude, probe_amplitude, 2)
    images = []

    for probe in probe_cube: 
        for amp in amps:
            surf_dm1 = dm1.getCurrentSurf()
            dm1.setSurf(surf_dm1+amp*probe)
            #surf_dm2 = dm2.getCurrentSurf()
            #dm2.setSurf(surf_dm2+amp*probe)
            image = get_photometry_2dm(context,navg,sleep_t=sleep_t)
            image = image[control_ports]
            images.append(image)
            dm1.setSurf(surf_dm1)
            #dm2.setSurf(surf_dm2)

    images = np.array(images)

    differential_images = differential_operator.dot(images)
    if return_all:
        return differential_images,images
    else:
        return differential_images
    
def calibrate(dm,probe_amplitude,probe_modes,calibration_amplitude,calibration_modes,navg=1,sleep_t=1e-12,control_ports=[0,1,2,3],perfect_lantern=False,coronagraph=False,extended_source=False):
    print('Calibrating iEFC...')
    Nmodes = calibration_modes.shape[0]
    
    response_matrix = []
    # Loop through all modes that you want to control
    start = time.time()
    for ci, calibration_mode in enumerate(calibration_modes):
        response = 0
        for s in [-1, 1]:
            # Set the DM to the correct state
            surf = dm.getCurrentSurf()
            dm.setSurf(surf + s * calibration_amplitude * calibration_mode)
            differential_images = take_measurement(dm, probe_modes, probe_amplitude,navg,sleep_t,control_ports=control_ports,perfect_lantern=perfect_lantern,coronagraph=coronagraph,extended_source=extended_source)
            response += s * differential_images / (2 * calibration_amplitude)
            dm.setSurf(surf)

        print(f'\tCalibrated mode {ci+1} / {Nmodes} in {time.time()-start:.2f}s', end='')
        print('\r', end='')
        
        measured_response = []
        for i in range(probe_modes.shape[0]):
            measured_response.append(response[i])
        measured_response = np.array(measured_response)
        response_matrix.append(np.concatenate(measured_response))

    print()
    print('Calibration complete.')
    
    response_matrix = np.array(response_matrix).T

    return response_matrix

def calibrate_2dm(dm1,dm2,probe_amplitude,probe_modes,calibration_amplitude,calibration_modes,navg=1,sleep_t=1e-12,control_ports=[0,1,2,3]):
    print('Calibrating iEFC...')
    Nmodes = calibration_modes.shape[0]

    response_matrix = []
    # Loop through all modes that you want to control
    start = time.time()

    #DM 1
    for ci, calibration_mode in enumerate(calibration_modes):
        response = 0
        for s in [-1, 1]:
            surf = dm1.getCurrentSurf()
            dm1.setSurf(surf + s * calibration_amplitude * calibration_mode)
            differential_images = take_measurement_2dm(dm1, dm2, probe_modes, probe_amplitude,navg,sleep_t,control_ports=control_ports)
            response += s * differential_images / (2 * calibration_amplitude)
            dm1.setSurf(surf)

        print(f'\tCalibrated mode {ci+1} / {Nmodes} in {time.time()-start:.2f}s', end='')
        print('\r', end='')
        
        measured_response = []
        for i in range(probe_modes.shape[0]):
            measured_response.append(response[i])
        measured_response = np.array(measured_response)
        response_matrix.append(np.concatenate(measured_response))

    #DM 2
    for ci, calibration_mode in enumerate(calibration_modes):
        response = 0
        for s in [-1, 1]:
            surf = dm2.getCurrentSurf()
            dm2.setSurf(surf + s * calibration_amplitude * calibration_mode)
            differential_images = take_measurement_2dm(dm1, dm2, probe_modes, probe_amplitude,navg,sleep_t,control_ports=control_ports)
            response += s * differential_images / (2 * calibration_amplitude)
            dm2.setSurf(surf)

        print(f'\tCalibrated mode {ci+1} / {Nmodes} in {time.time()-start:.2f}s', end='')
        print('\r', end='')
        
        measured_response = []
        for i in range(probe_modes.shape[0]):
            measured_response.append(response[i])
        measured_response = np.array(measured_response)
        response_matrix.append(np.concatenate(measured_response))

    print()
    print('Calibration complete.')
    
    response_matrix = np.array(response_matrix).T

    return response_matrix


def TikhonovInverse(A, rcond=1e-15):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_inv = s/(s**2 + (rcond * s.max())**2)
    return (Vt.T * s_inv).dot(U.T)

def beta_reg(S, beta=-1):
    # S is the sensitivity matrix also known as the Jacobian
    sts = np.matmul(S.T, S)
    rho = np.diag(sts)
    alpha2 = rho.max()

    gain_matrix = np.matmul( np.linalg.inv( sts + alpha2*10.0**(beta)*np.eye(sts.shape[0]) ), S.T)
    return gain_matrix

def single_iteration(dm,probe_cube,probe_amplitude,control_matrix,navg=1, sleep_t=1e-12,control_ports=[0,1,2,3],perfect_lantern=False,extended_source=False):
    # Take a measurement
    measurement_vector = take_measurement(dm, probe_cube, probe_amplitude,navg,sleep_t,control_ports,perfect_lantern=perfect_lantern,extended_source=extended_source).ravel()
    # Calculate the control signal in modal coefficients
    reconstructed_coefficients = control_matrix.dot(measurement_vector)
    return reconstructed_coefficients

def single_iteration_2dm(dm1,dm2,probe_cube,probe_amplitude,control_matrix,navg=1, sleep_t=1e-12,control_ports=[0,1,2,3]):
    # Take a measurement
    measurement_vector = take_measurement_2dm(dm1, dm2, probe_cube, probe_amplitude,navg,sleep_t,control_ports).ravel()
    # Calculate the control signal in modal coefficients
    reconstructed_coefficients = control_matrix.dot(measurement_vector)
    return reconstructed_coefficients
    
def run(dm, context,
        control_matrix,
        probe_modes, probe_amplitude,
        calibration_modes,
        num_iterations=10, 
        loop_gain=0.5, 
        leakage=0.0, navg=1,sleep_t=1e-12,control_ports=[0,1,2,3],perfect_lantern=False,extended_source=False):
    print('Running iEFC...')
    start = time.time()
    
    dm_commands = np.zeros((num_iterations+1, Nact, Nact), dtype=np.float64)
    metric_images = np.zeros((num_iterations+1, Nports), dtype=np.float64)
    command_coeffs = np.zeros((num_iterations+1,calibration_modes.shape[0]),dtype=np.float64)
    
    dm_ref = dm.getCurrentSurf()
    modal_coeff = 0.0

    dm_commands[0] = dm_ref
    metric_images[0] = get_photometry(context,navg, sleep_t=sleep_t,perfect_lantern=perfect_lantern,extended_source=extended_source)
    command_coeffs[0] = np.zeros(calibration_modes.shape[0])

    for i in range(num_iterations):
        #print(f"\tClosed-loop iteration {i+1} / {num_iterations}")
        
        delta_coefficients = single_iteration(dm, probe_modes, probe_amplitude, control_matrix, navg, sleep_t,control_ports,perfect_lantern=perfect_lantern,extended_source=extended_source)
        modal_coeff = (1.0-leakage)*modal_coeff + loop_gain*delta_coefficients
        command_coeffs[i+1] = modal_coeff
        
        # Reconstruct the full phase from the modes
        calibration_modes_reshaped = np.reshape(calibration_modes,(calibration_modes.shape[0],Nact**2))
        dm_command = -calibration_modes_reshaped.T.dot(modal_coeff).reshape(Nact,Nact)
        dm.setSurf(dm_ref + dm_command)
        
        # Take an image to estimate the metrics
        image = get_photometry(context,navg, sleep_t=sleep_t,perfect_lantern=perfect_lantern,extended_source=extended_source)
        
        metric_images[i+1] = image
        dm_commands[i+1] = dm.getCurrentSurf()
        
    print('iEFC loop completed in {:.3f}s.'.format(time.time()-start))
    return metric_images, dm_commands, command_coeffs

def run_2dm(dm1, dm2, context,
        control_matrix,
        probe_modes, probe_amplitude,
        calibration_modes,
        num_iterations=10, 
        loop_gain=0.5, 
        leakage=0.0, navg=1,sleep_t=1e-12,control_ports=[0,1,2,3]):
    print('Running iEFC...')
    start = time.time()
    
    dm1_commands = np.zeros((num_iterations+1, Nact, Nact), dtype=np.float64)
    dm2_commands = np.zeros((num_iterations+1, Nact, Nact), dtype=np.float64)
    metric_images = np.zeros((num_iterations+1, Nports), dtype=np.float64)
    command_coeffs = np.zeros((num_iterations+1,calibration_modes.shape[0]*2),dtype=np.float64)
    
    dm1_ref = dm1.getCurrentSurf()
    dm2_ref = dm2.getCurrentSurf()
    modal_coeff = 0.0

    dm1_commands[0] = dm1_ref
    dm2_commands[0] = dm2_ref
    metric_images[0] = get_photometry(context,navg, sleep_t=sleep_t)
    command_coeffs[0] = np.zeros(calibration_modes.shape[0]*2)

    for i in range(num_iterations):
        #print(f"\tClosed-loop iteration {i+1} / {num_iterations}")
        
        delta_coefficients = single_iteration_2dm(dm1, dm2, probe_modes, probe_amplitude, control_matrix, navg, sleep_t,control_ports)
        modal_coeff = (1.0-leakage)*modal_coeff + loop_gain*delta_coefficients
        command_coeffs[i+1] = modal_coeff

        #split coeffs into DM1 and DM2
        modal_coeff_1 = modal_coeff[0:calibration_modes.shape[0]]
        modal_coeff_2 = modal_coeff[calibration_modes.shape[0]:]
        
        # Reconstruct the full phase from the modes
        calibration_modes_reshaped = np.reshape(calibration_modes,(calibration_modes.shape[0],Nact**2))
        dm1_command = -calibration_modes_reshaped.T.dot(modal_coeff_1).reshape(Nact,Nact)
        dm1.setSurf(dm1_ref + dm1_command)
        dm2_command = -calibration_modes_reshaped.T.dot(modal_coeff_2).reshape(Nact,Nact)
        dm2.setSurf(dm2_ref + dm2_command)
        
        # Take an image to estimate the metrics
        image = get_photometry_2dm(context,navg, sleep_t=sleep_t)
        
        metric_images[i+1] = image
        dm1_commands[i+1] = dm1.getCurrentSurf()
        dm2_commands[i+1] = dm2.getCurrentSurf()
        
    print('iEFC loop completed in {:.3f}s.'.format(time.time()-start))
    return metric_images, dm1_commands, dm2_commands, command_coeffs
    
def scan_mode(dm,context,mode,amp,nsteps,navg=1,sleep_t=1e-12):
    amps = np.linspace(-amp,amp,nsteps)
    surf = dm.getCurrentSurf()
    fluxes = []
    for amp in amps:
        dm.setSurf(surf+amp*mode)
        fluxarray = get_photometry(context,navg,sleep_t=sleep_t)
        fluxes.append(fluxarray)
    fluxes = np.array(fluxes)
    dm.setSurf(surf)
    return fluxes

def scan_mode_diffims(dm,context,mode,amp,nsteps,probe_cube,probe_amplitude,navg=200,sleep_t=0.05):
    amps = np.linspace(-amp,amp,nsteps)
    surf = dm.getCurrentSurf()
    diffims = []
    for amp in amps:
        dm.setSurf(surf+amp*mode)
        diffim = take_measurement(dm, probe_cube, probe_amplitude,navg,sleep_t).ravel()
        diffims.append(diffim)
    diffims = np.array(diffims)
    dm.setSurf(surf)
    return diffims
    
#startup DM
dm = SimDM()

dm1 = SimDM()
dm2 = SimDM()

zernikes = create_zernike_modes(1,30)

##calibration modes

#zernikes
calibration_modes = zernikes[3:]
calibration_amplitude = 0.005*BMC_to_waves

nmodes = calibration_modes.shape[0]

##probe modes
probe_mode_1 = (zernikes[4]+zernikes[5]+zernikes[6]+zernikes[7])/np.sqrt(4)
probe_mode_2 = (zernikes[4]-zernikes[5]+zernikes[6]-zernikes[7])/np.sqrt(4)

#try random probes
#probe_mode_1 = np.random.normal(size=(Nact,Nact))
#pm1_normsq = np.sum(np.abs(np.square(probe_mode_1)))
#probe_mode_1 = probe_mode_1/np.sqrt(pm1_normsq)*12
#probe_mode_2 = np.random.normal(size=(Nact,Nact))
#pm2_normsq = np.sum(np.abs(np.square(probe_mode_2)))
#probe_mode_2 = probe_mode_2/np.sqrt(pm2_normsq)*12

probe_modes = np.array([probe_mode_1,probe_mode_2])
probe_amplitude = 0.01*BMC_to_waves