# Терминал и оболочка

> Терминал — место, где живут AI‑инженеры. Освойтесь здесь.

**Тип:** Learn  
**Языки:** --  
**Требования:** Phase 0, Lesson 01  
**Время:** ~35 минут

## Цели обучения

- Использовать пайпы, перенаправления и `grep` для фильтрации и обработки логов обучения из командной строки
- Создавать постоянные tmux‑сессии с несколькими панелями для одновременного запуска обучения и мониторинга GPU
- Мониторить системные ресурсы и GPU с помощью `htop`, `nvtop` и `nvidia-smi`
- Передавать файлы между локальной и удаленной машинами через SSH, `scp` и `rsync`

## Проблема

Вы будете проводить в терминале больше времени, чем в любом редакторе. Запуски обучения, мониторинг GPU, просмотр логов, удаленные SSH‑сессии, управление окружениями. Любой AI‑workflow так или иначе связан с оболочкой. Если вы медленно работаете в терминале — вы медленно работаете везде.

Этот урок охватывает навыки работы с терминалом, действительно важные для AI‑разработки. Без истории Unix. Без глубокого погружения в Bash‑скрипты. Только то, что действительно нужно.

## Концепция

```mermaid
graph TD
    subgraph tmux["tmux session: training"]
        subgraph top["Top row"]
            P1["Pane 1: Training run<br/>python train.py<br/>Epoch 12/100 ..."]
            P2["Pane 2: GPU monitor<br/>watch -n1 nvidia-smi<br/>GPU: 78% | Mem: 14/24G"]
        end
        P3["Pane 3: Logs + experiments<br/>tail -f logs/train.log | grep loss"]
    end
```

Три процесса одновременно. Один терминал. Вы можете отсоединиться, пойти домой, снова подключиться по SSH и продолжить работу. Обучение при этом не остановится.

## Практика

### Шаг 1: Узнайте свою оболочку

Проверьте, какую оболочку вы используете:

```bash
echo $SHELL
```

Большинство систем используют `bash` или `zsh`. Обе подходят. Команды из этого курса работают в любой из них.

Ключевые вещи, которые стоит знать:

```bash
# Move around
cd ~/projects/ai-engineering-from-scratch
pwd
ls -la

# History search (most useful shortcut you'll learn)
# Ctrl+R then type part of a previous command
# Press Ctrl+R again to cycle through matches

# Clear terminal
clear   # or Ctrl+L

# Cancel a running command
# Ctrl+C

# Suspend a running command (resume with fg)
# Ctrl+Z
```

### Шаг 2: Пайпы и перенаправления

Пайпы соединяют команды друг с другом. Именно так вы обрабатываете логи, фильтруете вывод и объединяете инструменты. Вы будете использовать это постоянно.

```bash
# Count how many times "loss" appears in a log
cat train.log | grep "loss" | wc -l

# Extract just the loss values from training output
grep "loss:" train.log | awk '{print $NF}' > losses.txt

# Watch a log file update in real time, filtering for errors
tail -f train.log | grep --line-buffered "ERROR"

# Sort experiments by final accuracy
grep "final_accuracy" results/*.log | sort -t= -k2 -n -r

# Redirect stdout and stderr to separate files
python train.py > output.log 2> errors.log

# Redirect both to the same file
python train.py > train_full.log 2>&1
```

Три перенаправления, которые вам действительно нужны:

| Символ | Что делает |
|--------|-------------|
| `>` | Записывает stdout в файл (с перезаписью) |
| `>>` | Добавляет stdout в конец файла |
| `2>` | Записывает stderr в файл |
| `2>&1` | Отправляет stderr туда же, куда и stdout |
| `\|` | Передает stdout одной команды как stdin следующей |

### Шаг 3: Фоновые процессы

Обучение моделей занимает часы. Вы не захотите держать терминал открытым все это время.

```bash
# Run in background (output still goes to terminal)
python train.py &

# Run in background, immune to hangup (closing terminal won't kill it)
nohup python train.py > train.log 2>&1 &

# Check what's running in background
jobs
ps aux | grep train.py

# Bring a background job to foreground
fg %1

# Kill a background process
kill %1
# or find its PID and kill that
kill $(pgrep -f "train.py")
```

Разница между `&`, `nohup` и `screen`/`tmux`:

| Метод | Переживает закрытие терминала? | Можно переподключиться? |
|--------|-------------------------------|--------------------------|
| `command &` | Нет | Нет |
| `nohup command &` | Да | Нет (смотрите лог-файл) |
| `screen` / `tmux` | Да | Да |

Для всего, что длится больше нескольких минут, используйте tmux.

### Шаг 4: tmux

tmux позволяет создавать постоянные терминальные сессии с несколькими панелями. Это один из самых полезных инструментов для управления обучением моделей.

```bash
# Install
# macOS
brew install tmux
# Ubuntu
sudo apt install tmux

# Start a named session
tmux new -s training

# Split horizontally
# Ctrl+B then "

# Split vertically
# Ctrl+B then %

# Navigate between panes
# Ctrl+B then arrow keys

# Detach (session keeps running)
# Ctrl+B then d

# Reattach
tmux attach -t training

# List sessions
tmux ls

# Kill a session
tmux kill-session -t training
```

Типичный AI‑workflow:

```bash
tmux new -s train

# Pane 1: start training
python train.py --epochs 100 --lr 1e-4

# Ctrl+B, " to split, then run GPU monitor
watch -n1 nvidia-smi

# Ctrl+B, % to split vertically, tail the logs
tail -f logs/experiment.log

# Now detach with Ctrl+B, d
# SSH out, go get coffee, come back
# tmux attach -t train
```

### Шаг 5: Мониторинг через htop и nvtop

```bash
# System processes (better than top)
htop

# GPU processes (if you have NVIDIA GPU)
# Install: sudo apt install nvtop (Ubuntu) or brew install nvtop (macOS)
nvtop

# Quick GPU check without nvtop
nvidia-smi

# Watch GPU usage update every second
watch -n1 nvidia-smi

# See which processes are using the GPU
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
```

Полезные клавиши в `htop`:
- `F6` или `>` — сортировка по столбцу
- `F5` — древовидный режим
- `F9` — завершить процесс
- `/` — поиск процесса

### Шаг 6: SSH для удаленных GPU‑серверов

Когда вы арендуете облачный GPU (Lambda, RunPod, Vast.ai), вы подключаетесь через SSH.

```bash
# Basic connection
ssh user@gpu-box-ip

# With a specific key
ssh -i ~/.ssh/my_gpu_key user@gpu-box-ip

# Copy files to remote
scp model.pt user@gpu-box-ip:~/models/

# Copy files from remote
scp user@gpu-box-ip:~/results/metrics.json ./

# Sync a whole directory (faster for many files)
rsync -avz ./data/ user@gpu-box-ip:~/data/

# Port forward (access remote Jupyter/TensorBoard locally)
ssh -L 8888:localhost:8888 user@gpu-box-ip
# Now open localhost:8888 in your browser

# SSH config for convenience
# Add to ~/.ssh/config:
# Host gpu
#     HostName 192.168.1.100
#     User ubuntu
#     IdentityFile ~/.ssh/gpu_key
#
# Then just:
# ssh gpu
```

### Шаг 7: Полезные алиасы для AI‑разработки

Добавьте это в `~/.bashrc` или `~/.zshrc`:

```bash
source phases/00-setup-and-tooling/10-terminal-and-shell/code/shell_aliases.sh
```

Или просто скопируйте нужные алиасы:

```bash
# GPU status at a glance
alias gpu='nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'

# Kill all Python training processes
alias killtraining='pkill -f "python.*train"'

# Quick virtual environment activate
alias ae='source .venv/bin/activate'

# Watch training loss
alias watchloss='tail -f logs/*.log | grep --line-buffered "loss"'
```

Полный список смотрите в `code/shell_aliases.sh`.

### Шаг 8: Частые terminal‑паттерны в AI

Эти команды постоянно встречаются на практике:

```bash
# Run training, log everything, notify when done
python train.py 2>&1 | tee train.log; echo "DONE" | mail -s "Training complete" you@email.com

# Compare two experiment logs side by side
diff <(grep "accuracy" exp1.log) <(grep "accuracy" exp2.log)

# Find the largest model files (clean up disk space)
find . -name "*.pt" -o -name "*.safetensors" | xargs du -h | sort -rh | head -20

# Download a model from Hugging Face
wget https://huggingface.co/model/resolve/main/model.safetensors

# Untar a dataset
tar xzf dataset.tar.gz -C ./data/

# Count lines in all Python files (see how big your project is)
find . -name "*.py" | xargs wc -l | tail -1

# Check disk space (training data fills disks fast)
df -h
du -sh ./data/*

# Environment variable check before training
env | grep -i cuda
env | grep -i torch
```

## Применение

Вот где именно инструменты пригодятся в этом курсе:

| Инструмент | Когда используется |
|------------|-------------------|
| tmux | Каждый запуск обучения (Phases 3+) |
| `tail -f` + `grep` | Мониторинг training‑логов |
| `nohup` / `&` | Быстрые фоновые задачи |
| `htop` / `nvtop` | Отладка медленного обучения и OOM‑ошибок |
| SSH + `rsync` | Работа с облачными GPU |
| Пайпы и перенаправления | Обработка результатов экспериментов |
| Алиасы | Экономия времени на повторяющихся командах |

## Упражнения

1. Установите tmux, создайте сессию с тремя панелями и запустите `htop` в одной, `watch -n1 date` во второй и Python‑скрипт в третьей. Отсоединитесь и подключитесь снова.
2. Добавьте алиасы из `code/shell_aliases.sh` в конфиг оболочки и перезагрузите его через `source ~/.zshrc` (или `~/.bashrc`).
3. Создайте фейковый training‑лог с `for i in $(seq 1 100); do echo "epoch $i loss: $(echo "scale=4; 1/$i" | bc)"; sleep 0.1; done > fake_train.log`, а затем используйте `grep`, `tail` и `awk`, чтобы извлечь только значения loss.
4. Настройте SSH config для сервера, к которому у вас есть доступ (или используйте `localhost` для практики).

## Ключевые термины

| Термин | Как обычно говорят | Что это означает на самом деле |
|--------|-------------------|--------------------------------|
| Shell | «Терминал» | Программа, интерпретирующая ваши команды (`bash`, `zsh`, `fish`) |
| tmux | «Терминальный мультиплексор» | Программа, позволяющая запускать несколько терминальных сессий в одном окне |
| Pipe | «Вертикальная черта» | Оператор `\|`, передающий вывод одной команды на вход другой |
| PID | «Идентификатор процесса» | Уникальный номер каждого процесса |
| nohup | «No hangup» | Запускает команду так, чтобы закрытие терминала ее не завершило |
| SSH | «Подключение к серверу» | Защищенный протокол для удаленного выполнения команд |
