from configparser import ConfigParser

#Get the configparser object
config_object = ConfigParser()

#Assume we need 2 sections in the config file, let's call them USERINFO and SERVERCONFIG
config_object["NOISE"] = {
    "AddNoise": "true",
    "MU": "0.",
    "THETA": "0.15",
    "SIGMA": "0.2"
}

config_object["DDPG"] = {
    "GAMMA" : "0.995",  #0.99                # Discount factor
    "TAU" : "1e-3"      # for soft update of target parameters 
}

config_object["AGENT"] = {
    "LR_ACTOR" : "1e-4",         # learning rate of the actor 
    "LR_CRITIC" : "1e-4",        # learning rate of the critic
    "WEIGHT_DECAY": "0.0",      # L2 weight decay
    "TRAIN_EVERY" :"20",        # How many iterations to wait before updating target networks
    "BUFFER_SIZE" : "1e6",
    "BATCH_SIZE": "128"
}

with open('config.ini', 'w') as conf:
    config_object.write(conf)