# DataSci 211 Tips

Development tips for GPU computing coursework on Marlowe.

## Using VSCode/Cursor with Marlowe via SSHFS

Marlowe blocks VSCode Remote-SSH server installations on login nodes. Use SSHFS to mount your remote directory locally and work with your IDE locally while maintaining terminal access to Marlowe.

### Setup

Add this function to your shell configuration (`~/.bashrc` or `~/.zshrc`):

```bash
mountviasshfs () {
    username=YOUR_SUNETID
    host="sherlock-home" ; if [ ! -z $1 ] ; then host=$1 ; fi
    if [ $host == "marlowe-home" ] ; then
        address=$username@login.marlowe.stanford.edu
        src="/users/$username" ; if [ ! -z $2 ] ; then src=$2 ; fi
    elif [ $host == "sherlock-home" ] ; then
        address=$username@dtn.sherlock.stanford.edu
        src="/home/users/$username$" ; if [ ! -z $2 ] ; then src=$2 ; fi
    fi
    dst=$host
    umount -f $HOME/$dst 2>/dev/null
    mkdir -p $HOME/$dst
    sshfs -o defer_permissions $address:$src/ $HOME/$dst
}
```

Replace `YOUR_SUNETID` with your SUNet ID.

### Usage

Mount your Marlowe home directory:
```bash
mountviasshfs marlowe-home
```

Open `~/marlowe-home/` in your IDE. File changes sync automatically. Use the IDE's integrated terminal to SSH into Marlowe for compilation and job submission.

### Requirements
- macOS: `brew install macfuse sshfs`
- Linux: `sudo apt install sshfs`
