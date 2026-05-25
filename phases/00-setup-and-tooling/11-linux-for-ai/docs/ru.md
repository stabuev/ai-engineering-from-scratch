# Linux для AI

> Большинство AI-систем работают на Linux. Вам нужно знать достаточно, чтобы не застрять.

**Тип:** Обучение  
**Языки:** --  
**Предварительные требования:** Фаза 0, Урок 01  
**Время:** ~30 минут

## Цели обучения

- Навигировать по файловой системе Linux и выполнять базовые операции с файлами из командной строки
- Управлять правами доступа с помощью `chmod` и `chown`, чтобы исправлять ошибки вида «Permission denied»
- Устанавливать системные пакеты через `apt` и настраивать свежий GPU-сервер для работы с AI
- Понимать различия между macOS и Linux, которые чаще всего мешают разработчикам при работе на удалённых машинах

## Проблема

Вы разрабатываете на macOS или Windows. Но как только вы подключаетесь по SSH к облачному GPU-серверу, арендуете инстанс Lambda или запускаете EC2-машину — вы оказываетесь в Ubuntu. Терминал становится вашим единственным интерфейсом. Нет ни Finder, ни Explorer, ни GUI. Если вы не умеете перемещаться по файловой системе, устанавливать пакеты и управлять процессами из командной строки, вы будете просто сжигать GPU-часы, гугля «как распаковать zip-файл в Linux».

Это руководство по выживанию. Здесь только то, что действительно нужно для работы с удалённой Linux-машиной в AI. Ничего лишнего.

## Структура файловой системы

Linux организует всё под единым корнем `/`. Здесь нет `C:\` или `/Volumes`. Каталоги, с которыми вы действительно будете работать:

```mermaid
graph TD
    root["/"] --> home["home/your-username/<br/>Your files — clone repos, run training"]
    root --> tmp["tmp/<br/>Temporary files, cleared on reboot"]
    root --> usr["usr/<br/>System programs and libraries"]
    root --> etc["etc/<br/>Config files"]
    root --> varlog["var/log/<br/>Logs — check when something breaks"]
    root --> mnt["mnt/ or /media/<br/>External drives and volumes"]
    root --> proc["proc/ and /sys/<br/>Virtual files — kernel and hardware info"]
```

Ваш домашний каталог — это `~` или `/home/your-username`. Почти всё, что вы делаете, происходит здесь.

## Основные команды

Это 15 команд, которые покрывают 95% того, что вы будете делать на удалённом GPU-сервере.

### Навигация

```bash
pwd                         # Where am I?
ls                          # What's here?
ls -la                      # What's here, including hidden files with details?
cd /path/to/dir             # Go there
cd ~                        # Go home
cd ..                       # Go up one level
```

### Файлы и директории

```bash
mkdir my-project            # Create a directory
mkdir -p a/b/c              # Create nested directories in one shot

cp file.txt backup.txt      # Copy a file
cp -r src/ src-backup/      # Copy a directory (recursive)

mv old.txt new.txt          # Rename a file
mv file.txt /tmp/           # Move a file

rm file.txt                 # Delete a file (no trash, it's gone)
rm -rf my-dir/              # Delete a directory and everything inside
```

`rm -rf` удаляет безвозвратно. Отмены нет. Перед нажатием Enter дважды проверьте путь.

### Чтение файлов

```bash
cat file.txt                # Print entire file
head -20 file.txt           # First 20 lines
tail -20 file.txt           # Last 20 lines
tail -f log.txt             # Follow a log file in real time (Ctrl+C to stop)
less file.txt               # Scroll through a file (q to quit)
```

### Поиск

```bash
grep "error" training.log           # Find lines containing "error"
grep -r "learning_rate" .           # Search all files in current directory
grep -i "cuda" config.yaml          # Case-insensitive search

find . -name "*.py"                 # Find all Python files under current dir
find . -name "*.ckpt" -size +1G     # Find checkpoint files larger than 1GB
```

## Права доступа

У каждого файла в Linux есть владелец и биты прав доступа. Вы столкнётесь с этим, когда скрипт не запускается или вы не можете записать файл в директорию.

```bash
ls -l train.py
# -rwxr-xr-- 1 user group 2048 Mar 19 10:00 train.py
#  ^^^             owner permissions: read, write, execute
#     ^^^          group permissions: read, execute
#        ^^        everyone else: read only
```

Частые исправления:

```bash
chmod +x train.sh           # Make a script executable
chmod 755 deploy.sh         # Owner: full, others: read+execute
chmod 644 config.yaml       # Owner: read+write, others: read only

chown user:group file.txt   # Change who owns a file (needs sudo)
```

Если вы видите «Permission denied», почти всегда проблема в правах доступа. В большинстве случаев помогут `chmod +x` или `sudo`.

## Управление пакетами (apt)

Ubuntu использует `apt`. Через него устанавливается системное ПО.

```bash
sudo apt update             # Refresh the package list (always do this first)
sudo apt install -y htop    # Install a package (-y skips confirmation)
sudo apt install -y build-essential  # C compiler, make, etc. Needed by many Python packages
sudo apt install -y tmux    # Terminal multiplexer (keep sessions alive after disconnect)

apt list --installed        # What's installed?
sudo apt remove htop        # Uninstall
```

Часто используемые пакеты для свежего GPU-сервера:

```bash
sudo apt update && sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    tmux \
    htop \
    unzip \
    python3-venv
```

## Пользователи и sudo

Обычно вы работаете как обычный пользователь. Некоторые операции требуют root-доступа.

```bash
whoami                      # What user am I?
sudo command                # Run a single command as root
sudo su                     # Become root (exit to go back, use sparingly)
```

На облачных GPU-инстансах вы чаще всего единственный пользователь и уже имеете доступ к sudo. Не запускайте всё от root. Используйте sudo только при необходимости.

## Процессы и systemd

Когда обучение зависло или нужно проверить, что сейчас работает:

```bash
htop                        # Interactive process viewer (q to quit)
ps aux | grep python        # Find running Python processes
kill 12345                  # Gracefully stop process with PID 12345
kill -9 12345               # Force kill (use when graceful doesn't work)
nvidia-smi                  # GPU processes and memory usage
```

systemd управляет сервисами (фоновыми демонами). Вы будете использовать его, если запускаете inference-серверы:

```bash
sudo systemctl start nginx          # Start a service
sudo systemctl stop nginx           # Stop it
sudo systemctl restart nginx        # Restart it
sudo systemctl status nginx         # Check if it's running
sudo systemctl enable nginx         # Start automatically on boot
```

## Дисковое пространство

GPU-серверы часто имеют ограниченный объём диска. Модели и датасеты быстро его заполняют.

```bash
df -h                       # Disk usage for all mounted drives
df -h /home                 # Disk usage for /home specifically

du -sh *                    # Size of each item in current directory
du -sh ~/.cache             # Size of your cache (pip, huggingface models land here)
du -sh /data/checkpoints/   # Check how big your checkpoints are

# Find the biggest space hogs
du -h --max-depth=1 / 2>/dev/null | sort -hr | head -20
```

Частые способы освободить место:

```bash
# Clear pip cache
pip cache purge

# Clear apt cache
sudo apt clean

# Remove old checkpoints you don't need
rm -rf checkpoints/epoch_01/ checkpoints/epoch_02/
```

## Сеть

Из командной строки вы будете скачивать модели, переносить файлы и обращаться к API.

```bash
# Download files
wget https://example.com/model.bin                   # Download a file
curl -O https://example.com/data.tar.gz              # Same thing with curl
curl -s https://api.example.com/health | python3 -m json.tool  # Hit an API, pretty-print JSON

# Transfer files between machines
scp model.bin user@remote:/data/                     # Copy file to remote machine
scp user@remote:/data/results.csv .                  # Copy file from remote to local
scp -r user@remote:/data/checkpoints/ ./local-dir/   # Copy directory

# Sync directories (faster than scp for large transfers, resumes on failure)
rsync -avz --progress ./data/ user@remote:/data/
rsync -avz --progress user@remote:/results/ ./results/
```

Для больших файлов используйте `rsync`, а не `scp`. Он передаёт только изменённые байты и умеет восстанавливаться после разрыва соединения.

## tmux: сохраняем сессии живыми

Если вы подключены по SSH к удалённому серверу, закрытие ноутбука завершит обучение. tmux предотвращает это.

```bash
tmux new -s train           # Start a new session named "train"
# ... start your training, then:
# Ctrl+B, then D            # Detach (training keeps running)

tmux ls                     # List sessions
tmux attach -t train        # Reattach to session

# Inside tmux:
# Ctrl+B, then %            # Split pane vertically
# Ctrl+B, then "            # Split pane horizontally
# Ctrl+B, then arrow keys   # Switch between panes
```

Всегда запускайте длительное обучение внутри tmux. Всегда.

## WSL2 для пользователей Windows

Если вы используете Windows, WSL2 даёт вам полноценную Linux-среду без dual boot.

```bash
# In PowerShell (admin)
wsl --install -d Ubuntu-24.04

# After restart, open Ubuntu from Start menu
sudo apt update && sudo apt upgrade -y
```

WSL2 запускает настоящее Linux-ядро. Всё из этого урока работает внутри него. Ваши Windows-файлы доступны по пути `/mnt/c/Users/YourName/` из WSL.

GPU passthrough работает с установленными NVIDIA-драйверами на стороне Windows. Установите Windows-драйвер NVIDIA (а не Linux-версию), и CUDA будет доступна внутри WSL2.

## Подводные камни: macOS → Linux

Что чаще всего вызывает проблемы у пользователей macOS:

| macOS | Linux | Комментарий |
|-------|-------|-------|
| `brew install` | `sudo apt install` | Названия пакетов иногда отличаются. `brew install htop` и `sudo apt install htop` работают одинаково, а вот `brew install readline` и `sudo apt install libreadline-dev` — нет. |
| `open file.txt` | `xdg-open file.txt` | Но на удалённом сервере GUI не будет. Используйте `cat` или `less`. |
| `pbcopy` / `pbpaste` | Нет аналога | Буфер обмена через пайпы не работает по SSH. |
| `~/.zshrc` | `~/.bashrc` | В macOS по умолчанию zsh. Большинство Linux-серверов используют bash. |
| `/opt/homebrew/` | `/usr/bin/`, `/usr/local/bin/` | Бинарники лежат в других местах. |
| `sed -i '' 's/a/b/' file` | `sed -i 's/a/b/' file` | В macOS sed требует пустую строку после `-i`, Linux — нет. |
| Файловая система без учёта регистра | Файловая система с учётом регистра | `Model.py` и `model.py` — два разных файла в Linux. |
| Концы строк `\n` | Концы строк `\n` | Здесь одинаково. Но Windows использует `\r\n`, что ломает bash-скрипты. Используйте `dos2unix`. |

## Краткая шпаргалка

```
Navigation:     pwd, ls, cd, find
Files:          cp, mv, rm, mkdir, cat, head, tail, less
Search:         grep, find
Permissions:    chmod, chown, sudo
Packages:       apt update, apt install
Processes:      htop, ps, kill, nvidia-smi
Services:       systemctl start/stop/restart/status
Disk:           df -h, du -sh
Network:        curl, wget, scp, rsync
Sessions:       tmux new/attach/detach
```

## Упражнения

1. Подключитесь к любой Linux-машине по SSH (или откройте WSL2), перейдите в домашнюю директорию. Создайте папку проекта, создайте внутри три пустых файла через `touch`, затем выведите их с помощью `ls -la`.
2. Установите `htop` через apt, запустите его и найдите процесс, который использует больше всего памяти.
3. Создайте tmux-сессию, выполните внутри `sleep 300`, отсоединитесь, выведите список сессий и подключитесь обратно.
4. Используйте `df -h` для проверки свободного места на диске, затем `du -sh ~/.cache/*`, чтобы найти, что занимает место в кэше.
5. Передайте файл с локальной машины на удалённую через `scp`, затем выполните ту же передачу через `rsync` и сравните опыт использования.
