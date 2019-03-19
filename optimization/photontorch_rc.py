''' Reservoir Computing for Photontorch

IMPORTANT:

This python file is meant to create an as close as possible reproduction of the caphe PhotonicSwirlNetwork.
It is used to prove that results of photontorch and caphe are equivalent.

However, it is a minimal implementation, and using photontorch should not be limited to only using this network.
On the contrary, the language of defining networks in photontorch is quite powerful and flexible. Please check it
out in the photontorch examples...

'''


#############
## Imports ##
#############

# Torch
import torch

# Other
import numpy as np

# Warnings
import warnings
warn = lambda s: warnings.warn(s, RuntimeWarning, 0)

# Photontorch
from photontorch import TwoPortNetwork # The photontorch equivalent of ConnMatrixNetwork
from photontorch.detectors import Photodetector
from photontorch.components import Waveguide, AgrawalSoa
from photontorch.environment import Environment



###########
## Swirl ##
###########

class PhotonicSwirlNetwork(TwoPortNetwork):
    def __init__(self,
        height=3,
        width=3,
        node_delay=1./160e9,
        signal_freq=160e9,
        loss_dB=3.0,
        sources_at = None,
        detectors_at = None,
        soas_at = None,
        random_phase=True,
        trainable=False,
        name=None):

        # fixed parameters
        self.ng=4.0
        self.prop_speed=299792458.0/self.ng
        self.wavelength = 1.55e-6

        # set parameters
        self.width = width
        self.height = height
        self.nr_nodes = nr_nodes = height*width
        self.internal_dt = self.external_dt = self.dt = 1/signal_freq
        self.delay = delay = node_delay
        loss_dB_m = 100*loss_dB
        self.inter_length = inter_length = self.prop_speed*delay
        self.attenuation_dB = loss_dB_m*inter_length

        # make connection matrix
        conn_matrix = self.get_conn_matrix()

        # create components
        phase = (lambda:0) if not random_phase else (lambda: 2*np.pi*np.random.rand())
        components = [
            Waveguide(
                length=inter_length,
                loss=loss_dB_m,
                neff=self.ng,
                phase=phase(),
                trainable=trainable, # if phase is trainable or not
                name='node%i'%i,
            )
        for i in range(nr_nodes)]
        
        if soas_at is not None:
            for i in soas_at:
                components[i] = AgrawalSoa(
                    L=inter_length,
                    neff=self.ng,
                    trainable=True,
                    name = 'soa%i'%i
                )

        # initialize
        if sources_at is None:
            sources_at = [0,16,32,48]
        print(f'sources_at={sources_at}')
        _sources_at = np.zeros(nr_nodes, bool)
        _sources_at[sources_at] = True
        
        if detectors_at is None:
            detectors_at = list(range(nr_nodes))
        print(f'detectors_at={detectors_at}')
        _detectors_at = np.zeros(nr_nodes, bool)
        _detectors_at[detectors_at] = True
        
        
        super(PhotonicSwirlNetwork, self).__init__(
            twoportcomponents=components,
            conn_matrix=conn_matrix,
            sources_at = _sources_at,
            detectors_at = _detectors_at,
            name=name,
        )

    def get_conn_matrix(self):
        # make connection matrix
        conn_matrix = np.complex64(swirl_topology_couplers(
            width=self.width,
            height=self.height,
            weight_north=0.5*0.5,
            weight_east=0.25*0.5,
            weight_south=0.25*0.5,
            weight_west=0.5*0.5,
        ))
        return conn_matrix
    

##############
## DC Swirl ##
##############

class PhotonicSwirlNetwork_DC(PhotonicSwirlNetwork):
    def get_conn_matrix(self):
        ## If we assume the delays between all components to be the same, we can just
        ## add conn_matrix and add_conn_matrix.
        conn_matrix, add_conn_matrix = dc_swirl_topology_couplers_with_connections(
            width=self.width,
            height=self.height,
            weight_north=0.5*0.5,
            weight_east=0.5*0.5,
            weight_south=0.5*0.5,
            weight_west=0.5*0.5,
        )
        return conn_matrix + add_conn_matrix



################
## Topologies ##
################

def swirl_topology_couplers(width, height, weight_north = 0.5*0.5, weight_east = 0.5*0.5, weight_south = 0.5*0.5, weight_west = 0.5*0.5):
	"""
	This function creates a swirl topology and adds specific weight values for the nearest
	neighbour connections in a certain direction from a node. It expects the weights to be
	power ratios but translates it into the weights for complex amplitudes.

	It works according to the physical interpretation: the rows are 'to', while the columns 'from'.

	The weights are according to the north/east/south/west direction of the connections and
	contain all the splitter and combiner ratios met in those directions (and also loss if wanted).

	The default values are for networks with 1x2 and 2x1 splitters everywhere. No splitters for
	branching off power for monitoring and reading out the reservoir are assumed present.

	The interpretation of the sequence of nodes should is row by row
	"""
	def getIndex(row_number, column_number, width):
		return row_number*width+column_number


	# create the connection matrix with
	nr_nodes = int(width*height)
	conn=np.zeros( (nr_nodes,nr_nodes), np.complex128)

	# loop over all the nodes
	for r in np.arange(height):
		for c in np.arange(width):

			# get the index of the node depending on its row and column
			index_node = getIndex(r, c, width)

			# check if the node is in the last column
			if(c+1 < width):
				# if not, get the index of the node in the next colunm
				index_east_node = getIndex(r, c+1, width)

				# check if the row number is in the upper part of the rectangular topology (and for an odd number of rows, do not include the center one)
				# if so, route to the east, else, route to the west between these two nodes
				if(r < int(height/2)):
					conn[index_east_node, index_node] = weight_east
				else:
					conn[index_node, index_east_node] = weight_west


			# check if the node is in the last row
			if(r+1 < height):
				# if not, get the index of the node in the next row
				index_south_node = getIndex(r+1, c, width)

				# check if the column number is in the left part of the rectangular topology (and for an odd number of columns, do not include the center one)
				# if so, route to the north, else, route to the south between these two nodes
				if(c < int(width/2)):
					conn[index_node,index_south_node] = weight_north
				else:
					conn[index_south_node,index_node] = weight_south

	# take the root of the weights
	conn = np.sqrt(conn)
	return conn

def dc_swirl_topology_couplers_with_connections(width, height, weight_north = 0.5*0.5, weight_east = 0.5*0.5, weight_south = 0.5*0.5, weight_west = 0.5*0.5):
    """
    
    Should only be used with even nr of rows and columns, otherwise directionality of the swirl is gone
    
    This function creates a swirl topology with directional coupler nodes and adds specific 
    weight values for the connections from a node (bidirectional). 
    
    It expects the weights to be 
	power ratios but translates it into the weights for complex amplitudes.
	
	It works according to the physical interpretation: the rows are 'from', while the columns 'to'. => i think also in previous topo??
	
	The weights are according to the north/east/south/west direction of the connections and 
	contain all the splitter and combiner ratios met in those directions (and also loss if wanted 
    - awaiting input from experiments for the dc instertion loss).
	
	The default values are for networks with 2x2 and 2x2 splitters everywhere, and connections around the
    reservoir as to be able to have a four-port everywhere (and keep directionality). No splitters for
	branching off power for monitoring and reading out the reservoir are assumed present in this level.
	
	The interpretation of the sequence of nodes is row by row"""
	
    def getIndex(row_number, column_number, width):
        return row_number*width+column_number

    # create the connection matrix with 
    nr_nodes = int((width)*(height))
    
    conn=np.zeros( (nr_nodes,nr_nodes), np.complex128)
    add_conn =np.zeros( (nr_nodes,nr_nodes), np.complex128)

    # loop over all the nodes
    for r in np.arange(height):
        for c in np.arange(width):
            # get the index of the node depending on its row and column
            index_node = getIndex(r, c, width)
            # check if the node is in the last column
            if(c+1 < width):
                # if not, get the index of the node in the next colunm
                index_east_node = getIndex(r, c+1, width)
                # check if the row number is in the upper part of the rectangular topology (and for an odd number of rows, do not include the center one)
                # if so, route to the east, else, route to the west between these two nodes
                if(r < int(height/2)):
                    conn[index_east_node, index_node] = weight_east
                    
                else:
                    conn[index_node, index_east_node] = weight_west
            # check if the node is in the last row
            if(r+1 < height):
                # if not, get the index of the node in the next row
                index_south_node = getIndex(r+1, c, width)
                # check if the column number is in the left part of the rectangular topology (and for an odd number of columns, do not include the center one)
                # if so, route to the north, else, route to the south between these two nodes
                if(c < int(width/2)):
                    conn[index_node,index_south_node] = weight_north
                else:
                    conn[index_south_node,index_node] = weight_south
                  
            if (r==0): 
                if (c==width/2-1):
                    add_conn[index_node+1,index_node]= weight_west

                elif (c>width/2-1):
                    index_mirror = getIndex(r, width-c-1, width)
                    conn[index_node,index_mirror] = weight_west
            
            if (r==height-1):
                if (c==width/2-1):   
                    add_conn[index_node,index_node+1]= weight_east

                elif (c<width/2-1):
                    index_mirror = getIndex(r, width-c-1, width)
                    conn[index_node,index_mirror] = weight_east
                    
            if (c==0):
                if (r==height/2-1):
                    add_conn[index_node,index_node+width]= weight_north
                    
                elif (r<height/2-1):
                    index_mirror = getIndex( height-r-1,c, width)
                    conn[index_node,index_mirror] = weight_north
                    
            if (c == width-1):
                if (r==height/2-1):
                    add_conn[index_node+width,index_node]= weight_south

                elif (r>height/2-1):
                    index_mirror = getIndex(height-r-1,c , width)
                    conn[index_node,index_mirror] = weight_south
                    # take the root of the weights
                    
    conn = np.sqrt(conn)
    add_conn = np.sqrt(add_conn)
    
    return conn, add_conn
