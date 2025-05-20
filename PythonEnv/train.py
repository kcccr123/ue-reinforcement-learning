import argparse as ap
import sys
from sb3_trainer import sb3_trainer

# -------------------- TRAINING SCRIPT -------------------- #

if __name__ == "__main__":

# ____Training Options ____ #
    parser = ap.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")

    args = parser.parse_args()
    if args.config is None:
        print("Path to config file not found.\n")
        sys.exit(1)

    sb3 = sb3_trainer(args.config)

    # Initialize and connect the AdminManager then set up env
    sb3.connect_bridge()

    # Create the model 
    sb3.init_model()

    # Train the model
    sb3.train_model()

    sb3.save_model()

    # TODO: Notify Unreal that training is complete through admin connection:
    # NEED TO ADD SEND COMMAND FOR ADMIN SOCKET, CURRENTLY USER JUST MANUALLY CLOSES UNREAL
    
    sb3.disconnect_bridge()
