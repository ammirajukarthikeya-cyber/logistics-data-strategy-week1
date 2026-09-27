"""
====================================================================================================
Week 1 Project: Strategic Planning and Data Exploration in Logistics
Module: logistics_strategy_pipeline.py
Description: End-to-end data pipeline demonstrating data generation, cleaning, KPI calculation,
             predictive delay modeling, spatial clustering, and heuristic route optimization.
====================================================================================================
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def generate_and_clean_logistics_data(n_samples: int = 2500, random_seed: int = 42) -> pd.DataFrame:
    """
    Simulates multi-echelon enterprise logistics transactional records and executes data hygiene.
    """
    np.random.seed(random_seed)
    
    order_ids = [f"ORD-{10000 + i}" for i in range(n_samples)]
    order_dates = [datetime(2026, 1, 1) + timedelta(days=float(np.random.uniform(0, 90))) for _ in range(n_samples)]
    distances_km = np.random.gamma(shape=2.5, scale=25.0, size=n_samples) + 2.0
    payload_weights_kg = np.random.uniform(1.0, 150.0, size=n_samples)
    shipping_modes = np.random.choice(["Standard", "Express", "Same-Day", "Freight"], size=n_samples, p=[0.55, 0.25, 0.12, 0.08])
    traffic_indices = np.random.uniform(1.0, 2.5, size=n_samples)
    
    speed_kmh = np.where(shipping_modes == "Same-Day", 35, np.where(shipping_modes == "Express", 50, 40))
    scheduled_hours = (distances_km / speed_kmh) * 1.2 + np.random.uniform(1, 4, size=n_samples)
    actual_hours = scheduled_hours * traffic_indices * np.random.uniform(0.9, 1.4, size=n_samples)
    
    df = pd.DataFrame({
        "order_id": order_ids,
        "order_timestamp": order_dates,
        "shipping_mode": shipping_modes,
        "distance_km": np.round(distances_km, 2),
        "payload_weight_kg": np.round(payload_weights_kg, 2),
        "traffic_index": np.round(traffic_indices, 2),
        "scheduled_transit_hours": np.round(scheduled_hours, 2),
        "actual_transit_hours": np.round(actual_hours, 2),
        "items_in_order": np.random.randint(1, 15, size=n_samples),
        "delivered_in_full": np.random.choice([1, 0], size=n_samples, p=[0.96, 0.04])
    })
    
    # Feature Derivations
    df["order_timestamp"] = pd.to_datetime(df["order_timestamp"])
    df["transit_delay_hours"] = df["actual_transit_hours"] - df["scheduled_transit_hours"]
    df["is_delayed"] = (df["transit_delay_hours"] > 0.5).astype(int)
    df["is_ontime_infull"] = ((df["is_delayed"] == 0) & (df["delivered_in_full"] == 1)).astype(int)
    
    return df


def compute_logistics_kpi_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes core supply chain KPIs (OTIF rate, delay variance, average transit times).
    """
    kpi_summary = df.groupby("shipping_mode").agg(
        total_orders=("order_id", "count"),
        avg_distance_km=("distance_km", "mean"),
        avg_actual_hours=("actual_transit_hours", "mean"),
        avg_delay_hours=("transit_delay_hours", "mean"),
        delay_variance=("transit_delay_hours", "var"),
        otif_rate_pct=("is_ontime_infull", lambda x: np.round(x.mean() * 100, 2)),
        sla_breach_rate_pct=("is_delayed", lambda x: np.round(x.mean() * 100, 2))
    ).reset_index()
    
    overall_otif = np.round(df["is_ontime_infull"].mean() * 100, 2)
    overall_delay = np.round(df["transit_delay_hours"].mean(), 2)
    
    print("\n" + "="*60)
    print("           NETWORK-WIDE KPI BENCHMARK DASHBOARD")
    print("="*60)
    print(f"Overall Network OTIF Rate:      {overall_otif}%")
    print(f"Average Transit Delay:          {overall_delay} hours")
    print("-" * 60)
    print(kpi_summary.to_string(index=False))
    print("="*60 + "\n")
    return kpi_summary


def build_predictive_delay_model(df: pd.DataFrame):
    """
    Trains a Machine Learning model (Random Forest) to forecast delivery delay risk prior to dispatch.
    """
    features = ["shipping_mode", "distance_km", "payload_weight_kg", "traffic_index", "items_in_order"]
    target = "is_delayed"
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), ["shipping_mode"]),
            ("num", "passthrough", ["distance_km", "payload_weight_kg", "traffic_index", "items_in_order"])
        ]
    )
    
    model_pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
    ])
    
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]
    
    print("="*60)
    print("       PREDICTIVE DELAY MODEL EVALUATION REPORT")
    print("="*60)
    print(f"Test Set ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print("-" * 60)
    print(classification_report(y_test, y_pred, digits=4))
    print("="*60 + "\n")
    return model_pipeline


def cluster_delivery_zones(n_locations: int = 500, n_clusters: int = 5) -> dict:
    """
    Partitions urban delivery coordinates into micro-fulfillment zones using K-Means.
    """
    np.random.seed(42)
    latitudes = 40.7128 + np.random.normal(0, 0.08, size=n_locations)
    longitudes = -74.0060 + np.random.normal(0, 0.08, size=n_locations)
    coords = np.column_stack((latitudes, longitudes))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(coords)
    hub_centroids = kmeans.cluster_centers_
    
    print("="*60)
    print(f"  SPATIAL CLUSTERING: {n_locations} STOPS -> {n_clusters} FULFILLMENT ZONES")
    print("="*60)
    for i, centroid in enumerate(hub_centroids):
        n_assigned = np.sum(cluster_labels == i)
        print(f"Zone {i+1}: Center (Lat: {centroid[0]:.4f}, Lon: {centroid[1]:.4f}) | Assigned Stops: {n_assigned}")
    print("="*60 + "\n")
    
    return {"coordinates": coords, "labels": cluster_labels, "centroids": hub_centroids}


def optimize_vehicle_route(stops_coords: np.ndarray, depot_idx: int = 0) -> tuple:
    """
    Executes a Traveling Salesperson Problem (TSP) heuristic for route sequencing.
    """
    n_stops = len(stops_coords)
    unvisited = set(range(n_stops))
    unvisited.remove(depot_idx)
    
    current_stop = depot_idx
    route = [current_stop]
    total_distance_km = 0.0
    
    dist_matrix = cdist(stops_coords, stops_coords, metric="euclidean") * 111.0
    
    while unvisited:
        next_stop = min(unvisited, key=lambda s: dist_matrix[current_stop][s])
        total_distance_km += dist_matrix[current_stop][next_stop]
        route.append(next_stop)
        unvisited.remove(next_stop)
        current_stop = next_stop
        
    total_distance_km += dist_matrix[current_stop][depot_idx]
    route.append(depot_idx)
    
    return route, total_distance_km


if __name__ == "__main__":
    print("Starting Week 1 Logistics Analytics Pipeline Execution...")
    df = generate_and_clean_logistics_data()
    kpis = compute_logistics_kpi_dashboard(df)
    model = build_predictive_delay_model(df)
    clusters = cluster_delivery_zones()
    
    sample_stops = clusters["coordinates"][:12]
    opt_route, dist = optimize_vehicle_route(sample_stops)
    print("="*60)
    print("         HEURISTIC VEHICLE ROUTE OPTIMIZATION")
    print("="*60)
    print(f"Optimized Stop Sequence: {' -> '.join(map(str, opt_route))}")
    print(f"Total Circuit Distance:  {dist:.2f} km")
    print("="*60)
    print("\nPipeline execution completed successfully!")
