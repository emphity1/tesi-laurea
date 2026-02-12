import time
import torch
import asyncio
import threading
import argparse
import traceback
import os
import sys
import bittensor as bt
from requests.exceptions import ConnectionError, InvalidSchema, RequestException
from src.base.neuron import BaseNeuron
from src.utils.config import add_miner_args
sys.path.insert(0, 'nsga-net')
from utilities import utils
import requests
import datetime as dt
from model.data import Model, ModelId
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
from model.storage.remote_model_store import RemoteModelStore
from model.dummy_trainer import DummyTrainer
from model.model_analysis import ModelAnalysis
from model.vali_config import ValidationConfig

class BaseMinerNeuron(BaseNeuron):
    neuron_type: str = "MinerNeuron"
    
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_miner_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        if not self.config.blacklist.force_validator_permit:
            bt.logging.warning(
                "You are allowing non-validators to send requests to your miner. This is a security risk."
            )
        if self.config.blacklist.allow_non_registered:
            bt.logging.warning(
                "You are allowing non-registered entities to send requests to your miner. This is a security risk."
            )

        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()
        self.save_dir = 'saved_model'

    
    async def run(self):
        self.sync()

        bt.logging.info(f"⛏️ Miner starting at block: {self.block}")
        try:
            vali_config = ValidationConfig()
            metadata_store = ChainModelMetadataStore(self.subtensor, self.wallet, self.config.netuid)
            remote_model_store = HuggingFaceModelStore()
            upload_dir = ""

            model_id = ModelId(namespace=self.config.hf_repo_id, name='naschain')
            HuggingFaceModelStore.assert_access_token_exists()

            if self.config.model.dir is None:
                bt.logging.info("Training Model!")
                trainer = DummyTrainer(epochs=vali_config.train_epochs)
                trainer.train()
                model = trainer.get_model()    
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                save_path = os.path.join(self.save_dir, 'model.pt')
                scripted_model = torch.jit.script(model)
                scripted_model.save(save_path)
                params = sum(param.numel() for param in model.parameters())
                bt.logging.info(f"🖥️ Params: {params}")
                upload_dir = save_path
                
            else:
                bt.logging.info("loading model offline!")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                try:
                    model = torch.jit.load(self.config.model.dir, map_location=device)
                    bt.logging.info("Torch script model loaded using torch.jit.load")
                except Exception as e:
                    bt.logging.warning(f"torch.jit.load failed with error: {e}")
                    try:
                        model = torch.load(self.config.model.dir,map_location="cpu")
                        bt.logging.info("Model loaded using torch.load")
                    except Exception as jit_e:
                        bt.logging.error(f"torch.load also failed with error: {jit_e}")
                        raise

                params = sum(param.numel() for param in model.parameters())
                bt.logging.info(f"🖥️ Params: {params}")
                upload_dir = self.config.model.dir

            model_id = await remote_model_store.upload_model(Model(id=model_id, pt_model=upload_dir))
            bt.logging.success(f"Uploaded model to hugging face. {model_id} , {upload_dir}")

            await metadata_store.store_model_metadata(
                self.wallet.hotkey.ss58_address, model_id)

            bt.logging.info(
                "Wrote model metadata to the chain. Checking we can read it back..."
            )

            model_metadata =  await metadata_store.retrieve_model_metadata(
                self.wallet.hotkey.ss58_address
            )
            bt.logging.info(f"⛏️ model_metadata: {model_metadata}")

            bt.logging.success("Committed model to the chain.")

            
        except Exception as e:
            bt.logging.error(f"Failed to advertise model on the chain: {e}")

    def run_in_background_thread(self):
        if not self.is_running:
            bt.logging.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run_async_main)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")
    
    def run_async_main(self):
        asyncio.run(self.run())

    def stop_run_thread(self):
        if self.is_running:
            bt.logging.debug("Stopping miner in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_run_thread()

    def resync_metagraph(self):
        bt.logging.info("resync_metagraph()")
        self.metagraph.sync(subtensor=self.subtensor)
