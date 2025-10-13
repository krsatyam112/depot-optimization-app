# depot_app.py
import streamlit as st
import pandas as pd
import numpy as np
from math import radians
from sklearn.cluster import KMeans, DBSCAN
import folium
from io import BytesIO
import base64
import tempfile
import os
from math import sin, cos, sqrt, atan2

EARTH_R_KM = 6371.0

# ---------- Utility functions ----------
def haversine_vec(lat1, lon1, lat2, lon2):
    lat1 = np.asarray(lat1)
    lon1 = np.asarray(lon1)
    lat2 = np.asarray(lat2)
    lon2 = np.asarray(lon2)
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_R_KM * c

def latlon_to_unit_sphere(lat, lon):
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    return np.vstack((x, y, z)).T

def xyz_to_latlon(xyz):
    x, y, z = xyz
    norm = np.sqrt(x*x + y*y + z*z)
    lat = np.degrees(np.arcsin(z / norm))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon

def assign_to_depots(latlon_array, depot_latlon_array, radius_km):
    if depot_latlon_array.size == 0:
        return np.full((len(latlon_array),), -1, dtype=int), np.full((len(latlon_array),), np.nan)
    cust_lat = latlon_array[:,0][:,None]
    cust_lon = latlon_array[:,1][:,None]
    depot_lat = depot_latlon_array[:,0][None,:]
    depot_lon = depot_latlon_array[:,1][None,:]
    dists = haversine_vec(cust_lat, cust_lon, depot_lat, depot_lon)  # (n,k)
    dists_masked = np.where(dists <= radius_km, dists, np.inf)
    nearest_idx = np.argmin(dists_masked, axis=1)
    nearest_dist = dists_masked[np.arange(dists_masked.shape[0]), nearest_idx]
    assigned = np.where(np.isfinite(nearest_dist), nearest_idx, -1)
    nearest_dist = np.where(np.isfinite(nearest_dist), nearest_dist, np.nan)
    return assigned, nearest_dist

def df_to_excel_bytes(df):
    out = BytesIO()
    df.to_excel(out, index=False, engine="openpyxl")
    out.seek(0)
    return out.read()

def folium_static_html(m):
    """Return HTML string of folium map for streamlit.components"""
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
    m.save(tmp.name)
    html = open(tmp.name, "r", encoding="utf-8").read()
    try:
        os.unlink(tmp.name)
    except:
        pass
    return html

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Depot Optimization App", layout="wide")
st.title("Depot Optimization — upload CSV to run analysis")

st.markdown(
    """
Upload a CSV with columns: **customer_geocode_lat**, **customer_geocode_long**, **Customer_code**.
Adjust parameters and click **Run analysis**.
"""
)

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
col1, col2, col3 = st.columns([1,1,1])

with col1:
    n_clusters = st.number_input("KMeans clusters (initial)", min_value=1, max_value=50, value=4, step=1)
with col2:
    max_radius = st.number_input("Feasible radius (km)", min_value=1.0, max_value=2000.0, value=100.0, step=1.0)
with col3:
    min_customers = st.number_input("Min customers to show depot", min_value=1, max_value=1000, value=15, step=1)

run_btn = st.button("Run analysis")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    required = ['customer_geocode_lat', 'customer_geocode_long', 'Customer_code']
    for c in required:
        if c not in df.columns:
            st.error(f"Missing required column: {c}")
            st.stop()

    df = df.dropna(subset=['customer_geocode_lat', 'customer_geocode_long']).reset_index(drop=True)
    X = df[['customer_geocode_lat', 'customer_geocode_long']].values

    if run_btn:
        st.info("Running clustering and refinement. This may take a few seconds...")
        # KMeans on sphere
        X_sphere = latlon_to_unit_sphere(X[:,0], X[:,1])
        km = KMeans(n_clusters=int(n_clusters), random_state=42, n_init=10)
        km.fit(X_sphere)
        centers_xyz = km.cluster_centers_
        initial_depots = np.array([xyz_to_latlon(c) for c in centers_xyz])

        st.write("Initial depot locations (approx):")
        for i, (lat, lon) in enumerate(initial_depots, 1):
            st.write(f"Depot {i}: {lat:.6f}, {lon:.6f}")

        # DBSCAN diagnostic
        X_rad = np.radians(X)
        dbscan_eps_km = min(20.0, max_radius/5.0)  # heuristic default
        db = DBSCAN(eps=dbscan_eps_km / EARTH_R_KM, min_samples=5, metric='haversine')
        labels = db.fit_predict(X_rad)
        n_clusters_db = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        st.write(f"DBSCAN (diagnostic) — clusters: {n_clusters_db}, noise: {n_noise}, eps_km ~ {dbscan_eps_km}")

        # Greedy assign to initial depots
        assigned_init, _ = assign_to_depots(X, initial_depots, max_radius)
        df['assigned_init'] = assigned_init

        # Refinement loop
        refined = initial_depots.copy()
        best_cov = np.sum(assign_to_depots(X, refined, max_radius)[0] != -1)
        st.write(f"Initial coverage: {best_cov}/{len(X)} customers")

        lr = 0.03
        max_iter = 100
        for it in range(max_iter):
            assigned, _ = assign_to_depots(X, refined, max_radius)
            moved = False
            new_ref = refined.copy()
            for idx in range(refined.shape[0]):
                cust_idx = np.where(assigned == idx)[0]
                if cust_idx.size == 0:
                    continue
                centroid = X[cust_idx].mean(axis=0)
                movement = (centroid - refined[idx]) * lr
                new_ref[idx] = refined[idx] + movement
            new_cov = np.sum(assign_to_depots(X, new_ref, max_radius)[0] != -1)
            if new_cov > best_cov:
                refined = new_ref
                best_cov = new_cov
                moved = True
            if not moved:
                break

        st.write(f"Final coverage after refinement: {best_cov}/{len(X)}")

        final_assigned, final_dist = assign_to_depots(X, refined, max_radius)
        df['depot_index'] = final_assigned
        df['Distance_from_Customer_km'] = final_dist
        df['Time_hr'] = df['Distance_from_Customer_km'] / 40.0
        df['Depot_Name'] = df['depot_index'].apply(lambda x: f"Depot_{int(x)+1}" if x != -1 else "Unassigned")
        dep_coords = np.array([ (np.nan, np.nan) if i==-1 else (refined[int(i)][0], refined[int(i)][1]) for i in df['depot_index'] ])
        df['Depot_Lat'] = dep_coords[:,0]
        df['Depot_Lon'] = dep_coords[:,1]

        # Per-depot counts
        depot_counts = df[df['depot_index']!=-1].groupby('depot_index').size().to_dict()
        st.write("Per-depot customer counts:")
        for k,v in sorted(depot_counts.items()):
            st.write(f"Depot {int(k)+1}: {v}")

        # Filter depots to display on map
        good_depots = [int(k) for k,v in depot_counts.items() if v >= int(min_customers)]
        filtered_depots = refined[good_depots] if len(good_depots)>0 else np.empty((0,2))

        # Output dataframe and download
        outcols = ['Customer_code','customer_geocode_lat','customer_geocode_long','depot_index','Depot_Name','Depot_Lat','Depot_Lon','Distance_from_Customer_km','Time_hr']
        out_df = df[outcols].copy()
        st.subheader("Result sample")
        st.dataframe(out_df.head(200))

        excel_bytes = df_to_excel_bytes(out_df)
        st.download_button(label="Download results (Excel)", data=excel_bytes, file_name="Customer_Depot_Analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Folium map
        st.subheader("Interactive map (Folium)")
        m_center = [df['customer_geocode_lat'].mean(), df['customer_geocode_long'].mean()]
        m = folium.Map(location=m_center, zoom_start=8)
        for _, r in df.iterrows():
            folium.CircleMarker(location=[r['customer_geocode_lat'], r['customer_geocode_long']],
                                radius=2, tooltip=str(r['Customer_code']), fill=True).add_to(m)
        for ridx in good_depots:
            lat, lon = refined[ridx]
            folium.Marker(location=[lat, lon], tooltip=f"Depot {ridx+1} ({depot_counts.get(ridx,0)} customers)").add_to(m)
            folium.Circle(location=[lat, lon], radius=max_radius*1000, fill=True, fill_opacity=0.06).add_to(m)

        # render folium map HTML
        map_html = folium_static_html(m)
        from streamlit.components.v1 import html as st_html
        st_html(map_html, height=600)

        st.success("Analysis complete. Use the download button to save results. Map is interactive.")
