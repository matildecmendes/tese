# Auditoria da adaptação LAND 3D → LAND-2D

## Conclusão

A adaptação foi implementada de forma paralela. Nenhum dos 46 ficheiros originais do ZIP foi modificado ou removido. A pipeline 3D original continua disponível exatamente como estava.

A nova pipeline é:

```text
.npy 2D normalizado em [-1,1]
        → VAE 2D
        → latentes 2D
        → LDM unconditional 2D
        → imagens sintéticas .npy (256,256), float32, [-1,1]
```

## Auditoria dos ficheiros originais

### Raiz

| Ficheiro | Decisão |
|---|---|
| `.gitignore` | Mantido sem alterações. |
| `LICENSE` | Mantido sem alterações. |
| `environment.yml` | Mantido sem alterações; contém as dependências usadas pela nova pipeline. |
| `readme.md` | Mantido como documentação oficial da versão 3D. |
| `launch.sh`, `launch2.sh`, `launchs.sh` | Mantidos; são launchers específicos da versão/infraestrutura original e não são usados pela pipeline 2D. |

### Scripts originais

| Ficheiro | Decisão |
|---|---|
| `scripts/preproc_data.sh` | Não usado: o dataset 2D já está preprocessado e normalizado. |
| `scripts/train_vae.sh` | Mantido para o VAE 3D original. |
| `scripts/inference_vae.sh` | Mantido para o VAE 3D original. |
| `scripts/train_vae_masks.sh` | Não usado: a adaptação é unconditional. |
| `scripts/inference_vae_masks.sh` | Não usado: não existe Mask VAE na adaptação. |
| `scripts/train_ldm.sh` | Mantido para o LDM 3D original. |
| `scripts/inference_ldm.sh` | Mantido para inferência 3D original. |

### Código original `src/`

| Ficheiro/pasta | Decisão |
|---|---|
| `src/train_ldm.py` | Não modificado; assume tensores 5D, volumes, máscaras e `dims=3`. |
| `src/inference_ldm.py` | Não modificado; orientado para volumes 3D e máscaras. |
| `src/pipeline/pipeline.py` | Não modificado; pipeline 3D/condicional original. |
| `src/unet/unet.py` | Não modificado, mas reutilizado pela classe `UNetModel2D`, porque a implementação aceita `dims=2`. |
| `src/utils/attention.py` | Não modificado; é utilizado indiretamente pelo U-Net original. |
| `src/utils/nn_utils.py` | Não modificado; é utilizado indiretamente pelo U-Net original. |
| `src/utils/opts.py` | Não utilizado pela nova CLI, para evitar argumentos 3D e de máscaras. |
| `src/utils/preproc_lidc_npy.py` | Não utilizado: parte de DICOM e cria volumes/máscaras. |
| `src/utils/utils_lidc3D.py` | Não utilizado: contém dataset, pipeline e utilitários volumétricos. |
| `src/utils/fix_pylidc_numpy.py` | Não utilizado: só é relevante para o preprocessing DICOM. |
| `src/vae/autoencoder_kl.py` | Não utilizado: a normalização MAISI personalizada exige tensores 5D. |
| `src/vae/vae_train.py` | Não utilizado: procura `chest_ct/*.npy`, aplica transforms/crops 3D e visualizações volumétricas. |
| `src/vae/vae_inference.py` | Não utilizado: inferência/reconstrução volumétrica. |
| `src/vae/vae_masks_train.py` | Não utilizado: treino do Mask VAE. |
| `src/vae/vae_masks_inference.py` | Não utilizado: inferência do Mask VAE. |
| `src/vae/configs/*` | Mantidos para as experiências 3D originais. |
| `src/vae/utils/losses.py` | Não modificado; a nova pipeline reutiliza apenas `KL_loss`. |
| `src/vae/utils/transforms.py` | Não utilizado; os transforms originais são volumétricos. |
| `src/vae/utils/masks_utils.py` | Não utilizado: não há conditioning por máscaras. |

## Ficheiros novos e localização

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

## Função de cada ficheiro novo

| Ficheiro | Função |
|---|---|
| `dataset_npy_2d.py` | Pesquisa recursiva, validação e carregamento de `.npy` 2D sem nova normalização. |
| `check_dataset_2d.py` | Verifica shapes, valores finitos e range do dataset antes do treino. |
| `autoencoder_kl_2d.py` | VAE do MONAI configurado para 2D e compatível com `save_pretrained/from_pretrained`. |
| `unet_2d.py` | Wrapper do U-Net LAND original com `dims=2` e sem cross-attention/máscaras. |
| `train_vae_2d.py` | Treino do VAE 2D com reconstrução, KL, perceptual e adversarial loss. |
| `inference_vae_2d.py` | Reconstrói slices reais e guarda comparações para validar o VAE. |
| `pipeline_ldm_2d.py` | Loop de denoising unconditional no espaço latente 2D e decoding. |
| `train_ldm_2d.py` | Treino do diffusion 2D com DDPM, `v_prediction`, SNR weighting, checkpoints e flip opcional (desativado por defeito). |
| `inference_ldm_2d.py` | Geração em batch e gravação de cada slice como `(256,256)` float32 em `[-1,1]`. |
| `config_vae_2d.json` | Arquitetura inicial do VAE 2D, alinhada com os canais/latentes do VAE original. |
| `config_vae_train_2d.json` | Hiperparâmetros iniciais do VAE, preservando os principais pesos do repo. |
| `scripts/*_2d.sh` | Interfaces simples que recebem os caminhos como argumentos. |

## Validações executadas

- Comparação SHA-256: 46/46 ficheiros originais inalterados.
- Compilação sintática de todos os módulos Python novos: concluída sem erros.
- Validação sintática (`bash -n`) dos cinco scripts Bash: concluída sem erros.
- Teste sintético do loader com os três formatos aceites:
  - `(256,256)`;
  - `(1,256,256)`;
  - `(256,256,1)`.
- Resultado do batch no teste: `(3,1,256,256)`, `torch.float32`, range `[-1,1]`.
- Testes de erro incluídos no código para shape incompatível, NaN/Inf, range inválido e incompatibilidade entre VAE/U-Net.

## Limite da validação neste ambiente

Não foi possível executar um treino real do VAE/LDM nem um forward completo com o ambiente oficial, porque este sandbox não contém MONAI, Diffusers e Accelerate e não disponibiliza a GPU/ambiente Conda do repositório. Por isso, o primeiro passo obrigatório no teu servidor é o smoke test documentado no README. Erros dependentes de CUDA, pesos opcionais do perceptual loss ou particularidades da instalação só podem ser confirmados nesse ambiente.
