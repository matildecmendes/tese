# LAND-2D — adaptação para slices 2D de Lung CT

Esta adaptação adiciona uma pipeline **2D, grayscale e unconditional** ao repositório LAND sem alterar os ficheiros 3D originais.


## Aplicar o patch no teu repositório Windows

No PowerShell, entra primeiro na raiz do teu repositório atual:

```powershell
cd C:\Users\matil.DESKTOP-6DVD4QL\Desktop\git_jobs\LAND
```

Coloca `LAND_2D_patch_only.zip` nessa pasta e executa:

```powershell
Expand-Archive -Path .\LAND_2D_patch_only.zip -DestinationPath .\ -Force
```

Confirma os novos ficheiros:

```powershell
Get-ChildItem .\src\land2d
Get-ChildItem .\scripts\*_2d.sh
git status
```

O ZIP do patch só acrescenta a implementação 2D e a documentação. Os comandos de treino usam scripts `.sh` e devem ser executados no servidor Linux/SLURM, tal como os scripts originais do LAND.

---

## Dados esperados

- Ficheiros `.npy` encontrados recursivamente dentro da pasta passada por argumento.
- Cada ficheiro pode ter shape `(256,256)`, `(1,256,256)` ou `(256,256,1)`.
- O loader devolve sempre tensores `(B,1,256,256)`.
- `dtype` final: `float32`.
- Valores já normalizados em `[-1,1]`.
- Não é aplicada uma segunda normalização.

## Ficheiros adicionados

```text
LAND/
├── README_LAND_2D_ADAPTATION.md
├── ADAPTATION_AUDIT.md
├── scripts/
│   ├── check_dataset_2d.sh
│   ├── train_vae_2d.sh
│   ├── inference_vae_2d.sh
│   ├── train_ldm_2d.sh
│   └── inference_ldm_2d.sh
└── src/land2d/
    ├── __init__.py
    ├── dataset_npy_2d.py
    ├── check_dataset_2d.py
    ├── autoencoder_kl_2d.py
    ├── unet_2d.py
    ├── train_vae_2d.py
    ├── inference_vae_2d.py
    ├── pipeline_ldm_2d.py
    ├── train_ldm_2d.py
    ├── inference_ldm_2d.py
    └── configs/
        ├── config_vae_2d.json
        └── config_vae_train_2d.json
```

Os seguintes ficheiros originais continuam intactos e não são usados pela pipeline 2D:

```text
scripts/preproc_data.sh
scripts/train_vae_masks.sh
scripts/inference_vae_masks.sh
src/utils/utils_lidc3D.py
src/vae/autoencoder_kl.py
src/vae/vae_train.py
src/train_ldm.py
src/inference_ldm.py
```

---

# Execução passo a passo

Todos os comandos seguintes devem ser executados a partir da pasta raiz `LAND` num ambiente Linux com o ambiente `land` ativado.

```bash
cd /caminho/para/LAND
conda activate land
```

Não é necessário executar `preproc_data.sh`, porque o dataset já está preprocessado.

## 1. Verificar o dataset

Primeiro inspeciona 1000 imagens:

```bash
bash scripts/check_dataset_2d.sh /caminho/para/LIDC_2D_NPY 1000
```

Para verificar todas as imagens:

```bash
bash scripts/check_dataset_2d.sh /caminho/para/LIDC_2D_NPY 0
```

O teste deve terminar sem erros e indicar:

```text
Expected shape: (256, 256)
Expected range: [-1, 1]
No invalid shapes, NaN/Inf values, or out-of-range values were found.
```

## 2. Smoke test do VAE

Executa apenas com 100 imagens, uma época e batch size 2:

```bash
bash scripts/train_vae_2d.sh \
  /caminho/para/LIDC_2D_NPY \
  vaeLAND2D_smoke \
  --max_samples 100 \
  --epochs 1 \
  --batch_size 2 \
  --val_batch_size 2 \
  --num_workers 0
```

O output será criado em:

```text
checkpoints/vae_2d/vaeLAND2D_smoke_<timestamp>/
```

Dentro dessa pasta devem aparecer:

```text
vae_best_epoch/
vae_last_epoch/
discriminator_best_epoch.pt
discriminator_last_epoch.pt
training_state_last.pt
tensorboard/
```

## 3. Treino completo do VAE

```bash
bash scripts/train_vae_2d.sh \
  /caminho/para/LIDC_2D_NPY \
  vaeLAND2D_LIDC
```

A configuração inicial é:

```text
resolution: 256
batch size: 8
latent channels: 4
VAE compression: 4x
learning rate: 1e-4
epochs: 100
reconstruction loss: L1
KL weight: 1e-7
perceptual weight: 0.3
adversarial weight: 0.1
```

Se a GPU não tiver memória suficiente:

```bash
bash scripts/train_vae_2d.sh \
  /caminho/para/LIDC_2D_NPY \
  vaeLAND2D_LIDC \
  --batch_size 4 \
  --val_batch_size 4
```

## 4. Continuar o treino do VAE

Usa o ficheiro `training_state_last.pt` do run:

```bash
bash scripts/train_vae_2d.sh \
  /caminho/para/LIDC_2D_NPY \
  vaeLAND2D_LIDC \
  --resume_from_checkpoint /caminho/para/training_state_last.pt
```

O argumento `run_name` é ignorado no destino quando se usa `--resume_from_checkpoint`; o treino continua na mesma pasta.

## 5. Validar as reconstruções do VAE

Substitui `VAE_PATH` pela pasta `vae_best_epoch`:

```bash
bash scripts/inference_vae_2d.sh \
  /caminho/para/LIDC_2D_NPY \
  /caminho/para/checkpoints/vae_2d/vaeLAND2D_LIDC_<timestamp>/vae_best_epoch \
  outputs/vae_2d_reconstructions \
  16
```

Abre os ficheiros:

```text
outputs/vae_2d_reconstructions/comparison_*.png
```

Antes de avançar para o LDM, confirma visualmente que:

- os pulmões mantêm a forma geral;
- não existe inversão de intensidades;
- não existem imagens completamente pretas ou brancas;
- não existem artefactos graves em grelha;
- as reconstruções têm anatomia reconhecível.

## 6. Smoke test do LDM

```bash
bash scripts/train_ldm_2d.sh \
  /caminho/para/LIDC_2D_NPY \
  /caminho/para/vae_best_epoch \
  LAND2D_LDM_smoke \
  --max_samples 100 \
  --num_epochs 1 \
  --max_train_steps 10 \
  --train_batch_size 2 \
  --dataloader_num_workers 0 \
  --save_model_epochs 1 \
  --save_images_epochs 1 \
  --sample_inference_steps 10
```

O output será criado em:

```text
checkpoints/ldm_2d/LAND2D_LDM_smoke/
├── unet/
├── scheduler/
├── resume_checkpoint/
├── training_samples/
└── args_2d.json
```

## 7. Treino completo do LDM

```bash
bash scripts/train_ldm_2d.sh \
  /caminho/para/LIDC_2D_NPY \
  /caminho/para/vae_best_epoch \
  LAND2D_LDM_LIDC
```

Configuração inicial:

```text
unconditional: sim
resolution: 256x256
latent resolution: 64x64
latent channels: 4
prediction: v_prediction
training diffusion steps: 1000
learning rate: 1e-5
batch size: 8
epochs: 500
SNR gamma: 5.0
attention: ativa
horizontal flip: desativado por defeito; usa `--horizontal_flip` apenas se o quiseres ativar
```

Para reduzir a memória:

```bash
bash scripts/train_ldm_2d.sh \
  /caminho/para/LIDC_2D_NPY \
  /caminho/para/vae_best_epoch \
  LAND2D_LDM_LIDC \
  --train_batch_size 4 \
  --gradient_accumulation_steps 2
```

Para usar mixed precision numa GPU compatível:

```bash
bash scripts/train_ldm_2d.sh \
  /caminho/para/LIDC_2D_NPY \
  /caminho/para/vae_best_epoch \
  LAND2D_LDM_LIDC \
  --mixed_precision fp16
```

## 8. Continuar o treino do LDM

```bash
bash scripts/train_ldm_2d.sh \
  /caminho/para/LIDC_2D_NPY \
  /caminho/para/vae_best_epoch \
  LAND2D_LDM_LIDC \
  --resume_from_checkpoint checkpoints/ldm_2d/LAND2D_LDM_LIDC/resume_checkpoint
```

O `resume_checkpoint` é substituído pela versão mais recente sempre que o modelo é guardado. Os snapshots adicionais de 50 em 50 épocas ficam em `epoch_ckpts/`.

## 9. Gerar 16 imagens de teste

```bash
bash scripts/inference_ldm_2d.sh \
  /caminho/para/vae_best_epoch \
  checkpoints/ldm_2d/LAND2D_LDM_LIDC \
  outputs/LAND2D_test \
  16 \
  1000 \
  4 \
  --save_png
```

Cada output `.npy` terá:

```text
shape: (256,256)
dtype: float32
range: [-1,1]
```

## 10. Gerar 5000 imagens

```bash
bash scripts/inference_ldm_2d.sh \
  /caminho/para/vae_best_epoch \
  checkpoints/ldm_2d/LAND2D_LDM_LIDC \
  outputs/LAND2D_5000 \
  5000 \
  1000 \
  8
```

Os nomes serão:

```text
image_00000.npy
image_00001.npy
...
image_04999.npy
```

A inferência ignora ficheiros já existentes. Cada índice usa a seed `42 + índice`. A lista completa de generators é também passada ao `DDPMScheduler`, mantendo cada amostra associada à mesma seed durante os passos de denoising.

---

# Utilização com SLURM

Os novos scripts não incluem recursos específicos do cluster, porque partição, memória, tempo e tipo de GPU dependem da tua infraestrutura. Podes submetê-los com os mesmos recursos que já usavas, por exemplo:

```bash
sbatch \
  --job-name=land2d_vae \
  --output=jobs_train/land2d_vae_%j.o \
  --error=jobs_train/land2d_vae_%j.e \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=48:00:00 \
  scripts/train_vae_2d.sh \
  /caminho/para/LIDC_2D_NPY \
  vaeLAND2D_LIDC
```

Adapta apenas os recursos e a partição às regras do teu cluster. O ambiente `land` deve estar disponível no job. Caso o cluster não exporte o ambiente Conda ativo, adiciona ao início do script as mesmas linhas `module load` e `conda activate` que já utilizas nos restantes treinos.

---

# Notas importantes

1. Não alteres `src/vae/autoencoder_kl.py`: continua a ser a implementação 3D original.
2. Não alteres `src/train_ldm.py`: continua a ser o treino 3D original.
3. Não uses `train_vae_masks.sh`: esta adaptação é unconditional.
4. Não uses `preproc_data.sh`: as tuas slices já estão preprocessadas.
5. Não voltes a normalizar o dataset.
6. Faz sempre os dois smoke tests antes do treino completo.
7. Só inicia o LDM depois de validar visualmente o VAE.
