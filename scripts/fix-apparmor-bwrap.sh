#!/bin/bash
# fix-apparmor-bwrap.sh
# Creates a targeted AppArmor profile for bubblewrap (bwrap) on Ubuntu 24.04+.
#
# This script satisfies the Ubuntu 24.04 requirement for a profile to be present
# to allow unprivileged user namespace creation, using a unified execution rule
# to avoid 'conflicting x modifiers' errors.

set -euo pipefail

if [ "$EUID" -ne 0 ]; then
    echo "Error: This script must be run as root (sudo)."
    exit 1
fi

BWRAP_PATH=$(which bwrap 2>/dev/null || echo "/usr/bin/bwrap")
BWRAP_REAL=$(readlink -f "$BWRAP_PATH")
PROFILE_FILE="/etc/apparmor.d/bwrap-sandbox"

echo "[*] bwrap binary: $BWRAP_PATH (resolved: $BWRAP_REAL)"

echo "[*] Creating AppArmor profile: $PROFILE_FILE"

cat > "$PROFILE_FILE" <<EOF
abi <abi/4.0>,
include <tunables/global>

profile bwrap-sandbox $BWRAP_REAL flags=(unconfined) {
  userns,

  # Allow bwrap to create and manage user namespaces
  capability setpcap,
  capability net_admin,
  capability sys_admin,
  capability sys_chroot,

  # Filesystem access
  / r,
  /** rwlkm,
  mount,
  umount,
  pivot_root,

  # Process and signal management
  signal,
  ptrace,
  unix,

  # Unified execution rule to prevent "conflicting x modifiers"
  /** pux,
}
EOF

echo "[+] Wrote profile to $PROFILE_FILE"

echo "[*] Loading profile..."
apparmor_parser -r "$PROFILE_FILE"
echo "[+] Profile loaded!"

echo "[*] Verifying bwrap works..."
if bwrap --unshare-user --unshare-pid --ro-bind / / --dev /dev --proc /proc -- echo "bwrap sandbox OK" 2>&1; then
    echo "[✓] Bubblewrap is now functional."
else
    echo ""
    echo "[!] bwrap-specific profile loaded but bwrap still failing."
    echo "    Fallback: sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0"
    exit 1
fi
