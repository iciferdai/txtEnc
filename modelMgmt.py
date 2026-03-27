import signal
from txtEncModel import *
from processData import *
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import time
import matplotlib.pyplot as plt
import chromadb
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

class ModelManagement():
    def __init__(self, model, train_dataloader, device=torch.device('cpu')):
        # === static ===
        self.EPOCH_PROGRESS_COUNT = 50
        self.EPOCH_CHECKPOINT_COUNT = 500
        self.EPOCH_IGNORE_CHECKPOINT = 50
        self.LR_PATIENCE = 100
        # === init ===
        self.model = model
        self.train_dl = train_dataloader
        self.device = device
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.vec_client = None
        # === dynamic related with model===
        self.train_loss = float('inf')
        self.best_train_loss = float('inf')
        # === dynamic related with mgmt===
        self.epoch_count = 0
        self.train_loss_list = []
        self.best_checkpoints = dict()
        # === tmp & plt & flags ===
        self.epoch = 0
        self.epochs = 0
        self.monitor_flag = []
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.train_line1 = None
        self.train_line2 = None
        # for manual exit
        self._register_signal_handler()

    def _register_signal_handler(self):
        #  SIGINT(Ctrl+C)   SIGTERM
        self.original_sigint = signal.getsignal(signal.SIGINT)
        self.original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handle_termination)
        signal.signal(signal.SIGTERM, self._handle_termination)

    def _handle_termination(self, signum, frame):
        print(f"\n!!! CATCH SIGNAL: {signum}, SAVING BEFORE TERMINATING!!!\n...")
        try:
            self.save_checkpoint()
            print("SAVED SUCCESS, EXIT NOW")
        except Exception as e:
            print(f"SAVED FAILED: {e}, EXIT")
        finally:
            signal.signal(signal.SIGINT, self.original_sigint)
            signal.signal(signal.SIGTERM, self.original_sigterm)
            exit(0)

    def init_weights(self):
        for name, param in self.model.named_parameters():
            if 'weight' in name and 'emb' in name:
                # 嵌入层初始化
                param.data.normal_(0.0, 1.0 / math.sqrt(D_MODEL))
            elif 'weight' in name and 'linear' in name:
                # 注意力层Linear
                if 'attn' in name:
                    nn.init.xavier_uniform_(param)
                # FFN层Linear
                elif 'ffn' in name:
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
            elif 'bias' in name:
                # 偏置初始化为0
                nn.init.zeros_(param)
            elif 'norm' in name:
                # 层归一化初始化为1（weight）和0（bias）
                if 'weight' in name:
                    nn.init.ones_(param)
                else:
                    nn.init.zeros_(param)

    def init_train(self):
        self.model.train()
        self.model.to(self.device)
        self.init_dashboard()
        # 分组设置weight_decay
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_ID)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=self.LR_PATIENCE, min_lr=1e-6)

    # 单样本做相似度损失，先不batch_size，（目前单词计算loss需要前向1+1+8=10次），输入要求：
    # anchor: [dim]
    # positive: [dim]
    # negative: [NEGATIVE_SAMPLE_NUM, dim]
    def contrast_loss(self, anchor, positive, negative, temperature=0.5):  # 按之前建议，温度先设0.5
        # 1. 归一化（保留）
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)

        # 2. 计算相似度
        # 锚点[1,1024]和正样本[1,1024]：结果是[1]
        pos_sim = torch.cosine_similarity(anchor, positive, dim=-1) / temperature

        # 先squeeze去掉anchor的冗余维度→[1024]，再unsqueeze(0)→[1,1024]，最后expand→[8,1024]
        anchor_expanded = anchor.expand(negative.shape[0], -1)  # [1,1024]→[8,1024]
        neg_sim = torch.cosine_similarity(anchor_expanded, negative, dim=-1) / temperature  # 结果是[8]

        # 3. 计算损失（保留逻辑，适配维度）
        pos_exp = torch.exp(pos_sim)  # [1]
        neg_exp_sum = torch.sum(torch.exp(neg_sim))  # 标量（8个负样本exp求和）
        loss = -torch.log(pos_exp / (pos_exp + neg_exp_sum)).mean()  # 加mean适配[1]的pos_exp

        return loss

    def init_sft_train(self, weight_path):
        self.model.train()
        self.load_checkpoint(weight_path, only_weights=True)
        for num, layer in enumerate(self.model.encoder_layers):
            for param in layer.parameters():
                param.requires_grad = False
            if num >= FROZE_LAYER_NUM-1: break
        for param in self.model.embedding.parameters():
            param.requires_grad = False

        self.model.to(self.device)
        self.init_dashboard()
        # 分组设置weight_decay
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=2e-5, weight_decay=0.01)
        self.criterion = self.contrast_loss
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=self.LR_PATIENCE, min_lr=1e-6)

    def init_eval(self):
        self.model.eval()
        self.model.to(self.device)

    def progress_info(self, force=False):
        if self.monitor_flag:
            logging.info(
                f"[{self.epoch + 1}/{self.epochs}]|Epoch_{self.epoch_count}] -> Loss: {self.train_loss:.4f}, "
                f"Best loss: {self.best_train_loss:.4f}, lr: {self.optimizer.param_groups[0]['lr']:.6f}; "
                f"\n ---> Monitor: {','.join(self.monitor_flag)}")
            self.monitor_flag = []
            self.update_dashboard()
        elif (self.epoch + 1) % self.EPOCH_PROGRESS_COUNT == 0:
            logging.info(
                f"[{self.epoch + 1}/{self.epochs}]|Epoch_{self.epoch_count}] -> Loss: {self.train_loss:.4f}, "
                f"Best loss: {self.best_train_loss:.4f}, lr: {self.optimizer.param_groups[0]['lr']:.6f}")
            self.update_dashboard()
        elif force:
            logging.info(
                f"[{self.epoch + 1}/{self.epochs}]|Epoch_{self.epoch_count}] -> Loss: {self.train_loss:.4f}, "
                f"Best loss: {self.best_train_loss:.4f}")
            logging.info(f'Best checkpoints: {self.best_checkpoints}')
            self.update_dashboard()

        if not force and (self.epoch_count % self.EPOCH_CHECKPOINT_COUNT == 0):
            self.save_checkpoint()

    def save_checkpoint(self, ckp_name=''):
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'train_loss': self.train_loss,
            'best_train_loss': self.best_train_loss
        }
        if ckp_name:
            weight_path = './saves/' + ckp_name
        else:
            weight_path = f'./saves/CheckPoint_Ep{self.epoch_count}_{self.train_loss:.4f}.pth'
        torch.save(checkpoint, weight_path)
        logging.info(f"checkpoint: {weight_path} Saved")

    def load_checkpoint(self, ckp_name='', only_weights=False):
        if not ckp_name:
            print('No checkpoint provided.')
            return
        weight_path = './saves/' + ckp_name
        try:
            ckpt = torch.load(weight_path, map_location=self.device)
            self.model.load_state_dict(ckpt["state_dict"])
            if not only_weights:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                self.scheduler.load_state_dict(ckpt["scheduler"])
                self.train_loss = ckpt["train_loss"]
                self.best_train_loss = ckpt["best_train_loss"]
            logging.info(f"checkpoint: {weight_path} Loaded")
        except Exception as e:
            logging.error(f"load_checkpoint Error: {e}", exc_info=True)

    def save_state(self, state_name=''):
        manager_state = {
            'epoch_count': self.epoch_count,
            'train_loss_list': self.train_loss_list,
            'best_checkpoints': self.best_checkpoints
        }
        if state_name:
            state_path = './saves/' + state_name
        else:
            state_path = f'./saves/State_Ep{self.epoch_count}_{self.best_train_loss:.4f}.pkl'
        with open(state_path, "wb") as f:
            pickle.dump(manager_state, f)
        logging.info(f"State saved at {state_path}")

    def load_state(self, state_name=''):
        if not state_name:
            print('No state provided.')
            return
        state_path = './saves/' + state_name
        try:
            with open(state_path, 'rb') as f:
                manager_state = pickle.load(f)
                self.epoch_count = manager_state['epoch_count']
                self.train_loss_list = manager_state['train_loss_list']
                self.best_checkpoints = manager_state['best_checkpoints']
                logging.info(f"State: {state_path} Loaded")
        except Exception as e:
            logging.error(f"load_state Error: {e}", exc_info=True)

    def clear_state(self):
        self.train_loss = float('inf')
        self.best_train_loss = float('inf')
        self.epoch_count = 0
        self.train_loss_list = []
        self.best_checkpoints = dict()

    def save_best(self):
        self.save_checkpoint('best_loss_cpt.pth')
        self.best_checkpoints[self.epoch_count] = self.train_loss
        self.save_state('best_state.pkl')

    def load_best(self):
        self.load_checkpoint('best_loss_cpt.pth', False)
        self.load_state('best_state.pkl')
        logging.info(f'Load Best; Best checkpoints: {self.best_checkpoints}')

    def roll_back(self, with_state=False):
        self.load_checkpoint('best_loss_cpt.pth', False)
        if with_state: self.load_state('best_state.pkl')

    def trans_data2dev(self, *args):
        transferred_args = []
        for arg in args:
            try:
                transferred_arg = arg.to(self.device, non_blocking=True)
                transferred_args.append(transferred_arg)
            except Exception as e:
                logging.error(f"trans_data2dev Error: {e}", exc_info=True)
                transferred_args.append(arg)

        if len(transferred_args) == 1:
            return transferred_args[0]
        return tuple(transferred_args)

    def init_dashboard(self):
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        # ion
        plt.switch_backend('TkAgg')
        plt.ion()
        # draw
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(20, 6), num="Loss Dashboard")
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Linear Scale')
        self.ax1.grid(alpha=0.2)
        self.ax2.set_yscale('log')
        self.ax2.set_xlabel('Epochs')
        self.ax2.set_ylabel('Loss')
        self.ax2.set_title('Log Scale')
        self.ax2.grid(alpha=0.2)
        # init
        self.train_line1, = self.ax1.plot(range(1, self.epoch_count+1),
                                        self.train_loss_list,
                                        label='Train Loss',
                                        marker='o',
                                        markersize=2)
        self.train_line2, = self.ax2.plot(range(1, self.epoch_count+1),
                                        self.train_loss_list,
                                        label='Train Loss',
                                        marker='o',
                                        markersize=2)
        # update
        self.ax1.legend()
        self.ax2.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_dashboard(self):
        self.train_line1.set_xdata(range(1, len(self.train_loss_list) + 1))
        self.train_line1.set_ydata(self.train_loss_list)
        self.train_line2.set_xdata(range(1, len(self.train_loss_list) + 1))
        self.train_line2.set_ydata(self.train_loss_list)
        # auto-set
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        # update
        self.ax1.legend()
        self.ax2.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # optional
        #time.sleep(0.01)

    def show_dashboard(self):
        plt.ioff()
        plt.show()

    def loss_algorithm(self):
        if self.train_loss < self.best_train_loss and self.epoch_count > self.EPOCH_IGNORE_CHECKPOINT:
            self.save_best()
            self.monitor_flag.append(f'Save Best Loss! ({self.best_train_loss:.4f}->{self.train_loss:.4f})')
            self.best_train_loss = self.train_loss

    def get_batch_loss(self, one_pack_data):
        src_mask, tgt_mask = generate_mlm_mask(one_pack_data[0])
        src_mask, tgt_mask = self.trans_data2dev(src_mask, tgt_mask)
        # forward
        output, vec, _ = self.model(src_mask)
        # loss
        output_flat = output.reshape(-1, VOCAB_SIZE)
        tgt_flat = tgt_mask.reshape(-1)
        loss = self.criterion(output_flat, tgt_flat)
        return loss

    def get_batch_output(self, one_pack_data):
        # unpack: Source -> processData
        src_mask, tgt_mask = generate_mlm_mask(one_pack_data[0])
        src_mask_gpu = self.trans_data2dev(src_mask)
        # forward
        output, vec, _ = self.model(src_mask_gpu)
        return output.cpu(), src_mask, tgt_mask

    def train_one_epoch(self):
        epoch_loss = torch.tensor(0.0, device=self.device)
        for one_pack_data in self.train_dl:
            self.optimizer.zero_grad()
            loss = self.get_batch_loss(one_pack_data)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss
        return epoch_loss.item()/len(self.train_dl)

    def train_epochs(self, eps):
        self.epochs = eps
        if self.optimizer is None:
            logging.error('NEED INIT TRAIN FIRST!!!')
            return

        for ep in range(self.epochs):
            self.epoch = ep
            self.train_loss = self.train_one_epoch()
            self.train_loss_list.append(self.train_loss)

            self.loss_algorithm()
            if self.scheduler is not None: self.scheduler.step(self.train_loss)
            self.epoch_count += 1
            self.progress_info()

        self.save_checkpoint()

    def get_one_sft_data(self, get_pos, get_neg):
        choice_neg_list = []
        for i in range(NEGATIVE_SAMPLE_NUM):
            choice_neg_list.append(random.choice(get_neg))
        return random.choice(get_pos), choice_neg_list

    def get_sft_loss(self, anchor, get_pos, get_neg):
        anchor, get_pos, get_neg = self.trans_data2dev(torch.tensor(anchor), torch.tensor(get_pos), torch.tensor(get_neg))
        # forward
        _, anchor_vec, _ = self.model(anchor)
        _, pos_vec, _ = self.model(get_pos)
        _, neg_vec, _ = self.model(get_neg)
        # loss
        loss = self.criterion(anchor_vec, pos_vec, neg_vec)
        return loss

    def train_one_sft(self):
        epoch_loss = torch.tensor(0.0, device=self.device)
        positive_list, negative_list, normal_list = self.train_dl
        for anchor in positive_list:
            get_pos, get_neg = self.get_one_sft_data(positive_list, negative_list+normal_list)
            self.optimizer.zero_grad()
            loss = self.get_sft_loss(anchor, get_pos, get_neg)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss
        for anchor in negative_list:
            get_pos, get_neg = self.get_one_sft_data(negative_list, positive_list + normal_list)
            self.optimizer.zero_grad()
            loss = self.get_sft_loss(anchor, get_pos, get_neg)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss
        for anchor in normal_list:
            get_pos, get_neg = self.get_one_sft_data(normal_list, positive_list + negative_list)
            self.optimizer.zero_grad()
            loss = self.get_sft_loss(anchor, get_pos, get_neg)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss
        return epoch_loss.item() / (len(positive_list)+len(negative_list)+len(normal_list))

    def train_sft_epochs(self, eps):
        self.epochs = eps
        if self.optimizer is None:
            logging.error('NEED INIT SFT TRAIN FIRST!!!')
            return

        for ep in range(self.epochs):
            self.epoch = ep
            self.train_loss = self.train_one_sft()
            self.train_loss_list.append(self.train_loss)

            self.loss_algorithm()
            if self.scheduler is not None: self.scheduler.step(self.train_loss)
            self.epoch_count += 1
            self.progress_info()

        self.save_checkpoint()

    def predict_auto(self):
        logging.info("Start auto testing...")
        for one_pack_data in self.train_dl:
            output, src_mask, tgt_mask = self.get_batch_output(one_pack_data)
            #output_flat = output.reshape(-1, VOCAB_SIZE)
            #tgt_flat = tgt_mask.reshape(-1)
            pred_token_ids = torch.argmax(output, dim=-1)
            for b in range(DEFAULT_BATCH_SIZE):
                pred_tokens = []
                src_mask_tokens = []
                tgt_tokens = []
                out_tokens = []
                for j, pred_id in enumerate(pred_token_ids[b]):
                    pred_token = idx2token[pred_id.item()]
                    src_id = src_mask[b][j].item()
                    src_token = idx2token[src_id]
                    if src_id == MASK_ID:
                        out_tokens.append(pred_token)
                        pred_tokens.append(pred_token)
                    elif src_id == PAD_ID:
                        continue
                    else:
                        out_tokens.append(src_token)
                        pred_tokens.append('-')

                    tgt_tokens.append(idx2token[one_pack_data[0][b][j].item()])
                    src_mask_tokens.append(src_token)

                print("====================== Target <-> Predicted ======================")
                print(f"{''.join(tgt_tokens)}")
                print(f"{''.join(out_tokens)}")
                print("======================= Mask <-> Predicted =======================")
                print(f"{''.join(src_mask_tokens)}")
                print(f"{''.join(pred_tokens)}")
                print("==================================================================")

    def init_qd_client(self):
        self.vec_client = QdrantClient(':memory:')
        self.vec_client.create_collection(
            collection_name='qd_vec_demo',
            vectors_config=VectorParams(size=D_MODEL, distance=Distance.COSINE))

    def init_ch_client(self):
        client = chromadb.Client(settings=chromadb.Settings(anonymized_telemetry=False))
        collection = client.get_or_create_collection(
            name="ch_vec_demo",
            metadata={"hnsw:space": "cosine"}  # 用余弦相似度计算
        )
        self.vec_client = collection

    def update_qd_client(self):
        points = []
        for i, (msg, _) in enumerate(demo_data):
            msg_ids = cov_ids(msg)
            msg_tensor = self.trans_data2dev(torch.tensor(msg_ids, dtype=torch.long))
            with torch.no_grad(): output, vec, _ = self.model(msg_tensor)
            points.append(PointStruct(id=i, vector=vec.cpu(), payload={'text': msg}))

        self.vec_client.upsert(collection_name='qd_vec_demo', points=points)
        logging.info(f'qd_client upsert {len(points)} points')

    def update_ch_client(self):
        id_batch, vec_batch, msg_batch =[], [], []
        for i, (msg, _) in enumerate(demo_data):
            msg_ids = cov_ids(msg)
            msg_tensor = self.trans_data2dev(torch.tensor(msg_ids, dtype=torch.long))
            with torch.no_grad(): output, vec, _ = self.model(msg_tensor)
            id_batch.append(str(i))
            msg_batch.append({'txt':msg})
            vec_batch.append(vec.cpu().tolist()[0])

        self.vec_client.add(ids=id_batch, embeddings=vec_batch, metadatas=msg_batch)
        logging.info(f'ch_client add {len(id_batch)} points')

    def qdrant_query(self, txt, res_num=3):
        input_ids = cov_ids(txt)
        msg_tensor = self.trans_data2dev(torch.tensor(input_ids, dtype=torch.long))
        with torch.no_grad(): output, vec, _ = self.model(msg_tensor)
        query_vec = vec.cpu().tolist()[0]

        results = self.vec_client.query_points(
            collection_name='qd_vec_demo',
            query=query_vec,
            limit=res_num,
            with_vectors=False,
            with_payload=True)

        print("======= Search Results =======")
        for i in results.points:
            print(f'id: {i.id}, vector: {i.vector}, payload: {i.payload}, score: {i.score}')

    def chroma_query(self, txt, res_num=3):
        input_ids = cov_ids(txt)
        msg_tensor = self.trans_data2dev(torch.tensor(input_ids, dtype=torch.long))
        with torch.no_grad(): output, vec, _ = self.model(msg_tensor)
        query_vec = vec.cpu().tolist()[0]

        results = self.vec_client.query(query_embeddings=query_vec, n_results=res_num)

        print("======= Search Results =======")
        res_ids=results['ids'][0]
        res_msg=results['metadatas'][0]
        res_dis=results['distances'][0]
        for i in range(len(res_ids)):
            print(f'id: {res_ids[i]}, distances: {res_dis[i]}, message: {res_msg[i]}')


if __name__ == '__main__':
    print('init model...')
    model = TxtEnc()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    d = process_data()
    print('init ModelManagement...')
    m_mgmt = ModelManagement(model, d, dev)
    m_mgmt.init_train()

    print('Empty, Do nothing, Exit...')