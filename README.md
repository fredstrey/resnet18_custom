# Coffee Leaf Disease Classifier

Interface web simples para classificação de doenças em folhas de café usando ResNet18.

## 🚀 Como usar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Iniciar o servidor API

Faça download dos pesos no link do drive: https://drive.google.com/file/d/1XVXXRdfn24tpPlNNZw2CaxJxpRs54WFY/view?usp=sharing e salve na pasta "model" do projeto.

Após isso, você poderá rodar o servidor com o comando:

```bash
python app.py
```

O servidor estará rodando em `http://localhost:8000`

### 3. Abrir a interface

Abra o arquivo `index.html` no seu navegador ou use um servidor local:

```bash
# Opção 1: Abrir diretamente
# Clique duas vezes em index.html

# Opção 2: Usar Python HTTP Server
python -m http.server 3000
# Acesse http://localhost:3000
```

## 📋 Funcionalidades

- ✅ Upload de imagens via clique ou drag & drop
- ✅ Preview da imagem enviada
- ✅ Exibição da classe predita
- ✅ Exibição da confiança com barra de progresso animada
- ✅ Interface moderna com tema escuro
- ✅ Animações suaves
- ✅ Responsivo para mobile

## 🎨 Interface

A interface possui:
- **Upload Area**: Área para fazer upload da imagem
- **Image Preview**: Visualização da imagem enviada
- **Classification Result**: Resultado da classificação com classe e confiança
- **Reset Button**: Botão para fazer upload de outra imagem

## 🔧 Tecnologias

- **Backend**: FastAPI + TensorFlow
- **Frontend**: HTML + CSS + JavaScript (Vanilla)
- **Modelo**: ResNet18 customizado
