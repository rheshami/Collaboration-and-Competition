from configparser import ConfigParser

#Read config.ini file
config_object = ConfigParser()
config_object.read("config.ini")


#agentConfig = config_object["AGENT"]

BUFFER_SIZE = config_object.getint("AGENT","buffer_size")   # replay buffer size
BATCH_SIZE = config_object.getint("AGENT","batch_size")                                # minibatch size
LR_ACTOR = config_object.getfloat("AGENT",'lr_actor')                                 # learning rate of the actor 
LR_CRITIC = config_object.getfloat("AGENT",'lr_critic')                              # learning rate of the critic
WEIGHT_DECAY = config_object.getfloat("AGENT",'weight_decay')                             # L2 weight decay
TRAIN_EVERY = config_object.getint("AGENT","train_every")                           # How many iterations to wait before updating target networks                          
LEARN_PER_EPISODE = config_object.getint("AGENT","learn_per_episode")


#ddpgConfig = config_object["DDPG"]
GAMMA = config_object.getfloat("DDPG","gamma")                                  # discount factor
TAU = config_object.getfloat("DDPG","tau")                                     # for soft update of target parameters
SEED = config_object.getint("DDPG","seed")

#noiseConfig = config_object["NOISE"]
ADD_NOISE = config_object.getboolean("NOISE","addnoise")
MU = config_object.getfloat("NOISE","mu")
THETA = config_object.getfloat("NOISE","theta")
SIGMA = config_object.getfloat("NOISE","sigma")
NOISE = config_object.getfloat("NOISE","noise")