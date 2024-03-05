Sub Main

Dim username, password
username = "hao"
password = "929910"

Dim Hosts(4)
hosts(0) = "192.168.3.5"
hosts(1) = "192.168.3.6"
hosts(2) = "192.168.3.9"
hosts(3) = "192.168.3.15"
hosts(4) = "192.168.3.16"


For Each HostStr In Hosts
    xsh.Session.Open ("ssh://" & username & ":" & password & "@" & HostStr)
    If xsh.Session.Connected Then
	End If
Next

End Sub
