FROM grafana/grafana:latest

ENV GF_INSTALL_PLUGINS=grafana-piechart-panel
# No admin password in the image; pass GF_SECURITY_ADMIN_PASSWORD at run time.

COPY monitoring/grafana/provisioning /etc/grafana/provisioning