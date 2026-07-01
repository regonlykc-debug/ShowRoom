# Run as Administrator in PowerShell
# Forwards both ports the tablet needs: 8080 (the catalogue site) and 8000 (the AI server).

$wslIp = wsl hostname -I | ForEach-Object { $_.Trim().Split(' ')[0] }
Write-Host "WSL2 IP: $wslIp"

foreach ($port in 8080, 8000) {
    netsh interface portproxy delete v4tov4 listenport=$port listenaddress=0.0.0.0 2>$null
    netsh interface portproxy add v4tov4 listenport=$port listenaddress=0.0.0.0 connectport=$port connectaddress=$wslIp
}

netsh advfirewall firewall delete rule name="ShowRoom Site" 2>$null
netsh advfirewall firewall add rule name="ShowRoom Site" dir=in action=allow protocol=TCP localport=8080

netsh advfirewall firewall delete rule name="ShowRoom AI" 2>$null
netsh advfirewall firewall add rule name="ShowRoom AI" dir=in action=allow protocol=TCP localport=8000

Write-Host ""
Write-Host "Done! On the tablet, open: http://192.168.100.62:8080"
Write-Host ""
Write-Host "Forwarding: 192.168.100.62:8080 -> ${wslIp}:8080 (site)"
Write-Host "Forwarding: 192.168.100.62:8000 -> ${wslIp}:8000 (AI)"
