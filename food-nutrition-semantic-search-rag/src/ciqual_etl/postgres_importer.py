#!/usr/bin/env python3
"""PostgreSQL importer with staging, validation, and reporting."""

import os
import sys
import logging
from collections import Counter
from typing import List, Set, Tuple, Optional

import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ciqual_etl import DatabaseConnection, FoodGroup, Food, Component, Composition, DataSource

logger = logging.getLogger(__name__)

class PostgresImporter:
    """
    Handles importing Ciqual data into PostgreSQL.

    This class manages database connections, creates staging and final tables,
    inserts data in batches, validates foreign key references, logs orphaned
    records, and generates reconciliation reports with charts.
    """
      
    def __init__(self):
        """Initialize the importer with no active database connection."""
        self.db = DatabaseConnection(exit_on_failure=True)
        self.conn = None
        self.cur = None
        
    def connect(self) -> None:
        self.db.connect()
        self.conn = self.db.conn
        self.cur = self.db.cur

    def disconnect(self) -> None:
        self.db.disconnect()

    def create_staging_tables(self) -> None:
        """
        Create staging tables that mirror the XML structure.

        Staging tables have no foreign key constraints, allowing raw ingestion
        of all data regardless of referential integrity. This enables later
        validation and cleaning.
        """
        
        logger.info("Creating staging tables...")
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS staging_food_groups (
                alim_grp_code TEXT, alim_ssgrp_code TEXT, alim_ssssgrp_code TEXT,
                alim_grp_nom_eng TEXT, alim_ssgrp_nom_eng TEXT, alim_ssssgrp_nom_eng TEXT,
                alim_grp_nom_fr TEXT, alim_ssgrp_nom_fr TEXT, alim_ssssgrp_nom_fr TEXT
            )
        """)
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS staging_foods (
                alim_code INTEGER, alim_nom_eng TEXT, alim_nom_fr TEXT, alim_nom_sci TEXT,
                facteur_jones NUMERIC(5,3), alim_grp_code TEXT, alim_ssgrp_code TEXT, alim_ssssgrp_code TEXT
            )
        """)
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS staging_components (
                const_code INTEGER, const_nom_eng TEXT, const_nom_fr TEXT, code_infoods TEXT
            )
        """)
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS staging_data_sources (
                source_code INTEGER, ref_citation TEXT
            )
        """)
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS staging_composition (
                alim_code INTEGER, const_code INTEGER, teneur TEXT, min_val TEXT,
                max_val TEXT, code_confiance CHAR(1), source_code INTEGER
            )
        """)
        self.conn.commit()
        logger.info("Staging tables created")

    def create_log_tables(self) -> None:
        """Create tables for logging orphaned records (foods and compositions)."""
        
        logger.info("Creating log tables if they don't exist...")
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS orphan_foods (
                alim_code INTEGER, reason TEXT, created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS orphan_compositions (
                alim_code INTEGER, const_code INTEGER, reason TEXT, created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        self.conn.commit()
        logger.info("Logging tables created")

    def create_tables(self) -> None:
        """
        Create the final normalized tables with primary and foreign keys.

        Also creates performance indexes on foreign key columns.
        Tables are created if they do not already exist.
        """
        
        logger.info("Creating tables if they don't exist...")
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS food_groups (
                alim_grp_code TEXT NOT NULL, alim_ssgrp_code TEXT NOT NULL, alim_ssssgrp_code TEXT NOT NULL,
                alim_grp_nom_eng TEXT, alim_ssgrp_nom_eng TEXT, alim_ssssgrp_nom_eng TEXT,
                alim_grp_nom_fr TEXT, alim_ssgrp_nom_fr TEXT, alim_ssssgrp_nom_fr TEXT,
                PRIMARY KEY (alim_grp_code, alim_ssgrp_code, alim_ssssgrp_code)
            )
        """)
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS foods (
                alim_code INTEGER PRIMARY KEY, alim_nom_eng TEXT, alim_nom_fr TEXT, alim_nom_sci TEXT,
                facteur_jones NUMERIC(5,3), alim_grp_code TEXT, alim_ssgrp_code TEXT, alim_ssssgrp_code TEXT,
                image_front_url TEXT, image_small_url TEXT, image_thumb_url TEXT,
                off_product_name TEXT, off_brand TEXT, off_last_updated TIMESTAMP,
                FOREIGN KEY (alim_grp_code, alim_ssgrp_code, alim_ssssgrp_code)
                REFERENCES food_groups(alim_grp_code, alim_ssgrp_code, alim_ssssgrp_code)
            )
        """)
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS components (
                const_code INTEGER PRIMARY KEY, const_nom_eng TEXT, const_nom_fr TEXT, code_infoods TEXT
            )
        """)
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS data_sources (
                source_code INTEGER PRIMARY KEY, ref_citation TEXT
            )
        """)
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS composition (
                id SERIAL PRIMARY KEY, alim_code INTEGER NOT NULL, const_code INTEGER NOT NULL,
                teneur TEXT, min_val TEXT, max_val TEXT, code_confiance CHAR(1), source_code INTEGER,
                FOREIGN KEY (alim_code) REFERENCES foods(alim_code),
                FOREIGN KEY (const_code) REFERENCES components(const_code),
                FOREIGN KEY (source_code) REFERENCES data_sources(source_code)
            )
        """)
        # Indexes
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_composition_alim ON composition(alim_code)")
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_composition_const ON composition(const_code)")
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_foods_group ON foods(alim_grp_code, alim_ssgrp_code, alim_ssssgrp_code)")
        self.conn.commit()
        logger.info("Clean tables created")

    def clear_tables(self) -> None:
        """
        Clear all existing data from staging, logging, and final tables.

        Useful for performing a fresh import. Uses TRUNCATE ... CASCADE to
        handle foreign key dependencies gracefully.
        """
        
        logger.info("Clearing existing data from all tables...")
        tables = ["staging_composition", "staging_foods", "staging_food_groups",
                  "staging_components", "staging_data_sources", "orphan_compositions",
                  "orphan_foods", "composition", "foods", "food_groups",
                  "components", "data_sources"]
        for t in tables:
            self.cur.execute(f"TRUNCATE TABLE {t} CASCADE")
        self.conn.commit()
        logger.info("Tables cleared")

    # ---- Staging inserts ----
    def insert_staging_food_groups(self, food_groups: List[FoodGroup]) -> None:
        """
        Insert raw food group data into the staging_food_groups table.

        Args:
            food_groups (List[FoodGroup]): List of FoodGroup objects to insert.
        """
        
        if not food_groups:
            return
        data = [(fg.alim_grp_code, fg.alim_ssgrp_code, fg.alim_ssssgrp_code,
                 fg.alim_grp_nom_eng, fg.alim_ssgrp_nom_eng, fg.alim_ssssgrp_nom_eng,
                 fg.alim_grp_nom_fr, fg.alim_ssgrp_nom_fr, fg.alim_ssssgrp_nom_fr)
                for fg in food_groups]
        execute_values(self.cur, """
            INSERT INTO staging_food_groups
            (alim_grp_code, alim_ssgrp_code, alim_ssssgrp_code,
             alim_grp_nom_eng, alim_ssgrp_nom_eng, alim_ssssgrp_nom_eng,
             alim_grp_nom_fr, alim_ssgrp_nom_fr, alim_ssssgrp_nom_fr)
            VALUES %s
        """, data)
        self.conn.commit()
        logger.info(f"Inserted {len(food_groups)} food groups into staging")

    def insert_staging_foods(self, foods: List[Food]) -> None:
        """
        Insert raw food data into the staging_foods table.

        Args:
            foods (List[Food]): List of Food objects to insert.
        """
        
        if not foods:
            return
        data = [(f.alim_code, f.alim_nom_eng, f.alim_nom_fr, f.alim_nom_sci,
                 f.facteur_jones, f.alim_grp_code, f.alim_ssgrp_code, f.alim_ssssgrp_code)
                for f in foods]
        execute_values(self.cur, """
            INSERT INTO staging_foods
            (alim_code, alim_nom_eng, alim_nom_fr, alim_nom_sci,
             facteur_jones, alim_grp_code, alim_ssgrp_code, alim_ssssgrp_code)
            VALUES %s
        """, data)
        self.conn.commit()
        logger.info(f"Inserted {len(foods)} foods into staging")

    def insert_staging_components(self, components: List[Component]) -> None:
        """
        Insert raw component data into the staging_components table.

        Args:
            components (List[Component]): List of Component objects to insert.
        """
        
        if not components:
            return
        data = [(c.const_code, c.const_nom_eng, c.const_nom_fr, c.code_infoods) for c in components]
        execute_values(self.cur, """
            INSERT INTO staging_components (const_code, const_nom_eng, const_nom_fr, code_infoods)
            VALUES %s
        """, data)
        self.conn.commit()
        logger.info(f"Inserted {len(components)} components into staging")

    def insert_staging_composition(self, compositions: List[Composition]) -> None:
        """
        Insert raw composition data into the staging_composition table.

        Args:
            compositions (List[Composition]): List of Composition objects to insert.
        """
        
        if not compositions:
            return
        data = [(c.alim_code, c.const_code, c.teneur, c.min_val,
                 c.max_val, c.code_confiance, c.source_code) for c in compositions]
        execute_values(self.cur, """
            INSERT INTO staging_composition
            (alim_code, const_code, teneur, min_val, max_val, code_confiance, source_code)
            VALUES %s
        """, data)
        self.conn.commit()
        logger.info(f"Inserted {len(compositions)} composition records into staging")

    def insert_staging_data_sources(self, sources: List[DataSource]) -> None:
        """
        Insert raw data source references into the staging_data_sources table.

        Args:
            sources (List[DataSource]): List of DataSource objects to insert.
        """
        
        if not sources:
            return
        data = [(s.source_code, s.ref_citation) for s in sources]
        execute_values(self.cur, """
            INSERT INTO staging_data_sources (source_code, ref_citation) VALUES %s
        """, data)
        self.conn.commit()
        logger.info(f"Inserted {len(sources)} data sources into staging")

    # ---- Clean inserts with validation ----
    def insert_unknown_food_groups(self) -> None:
        """
        Insert a special 'UNKNOWN' food group into the clean food_groups table.

        This group absorbs placeholder or missing group codes during validation.
        The operation is idempotent (ON CONFLICT DO NOTHING).
        """
        
        self.cur.execute("""
            INSERT INTO food_groups (alim_grp_code, alim_ssgrp_code, alim_ssssgrp_code,
             alim_grp_nom_eng, alim_ssgrp_nom_eng, alim_ssssgrp_nom_eng,
             alim_grp_nom_fr, alim_ssgrp_nom_fr, alim_ssssgrp_nom_fr)
            VALUES ('UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'Unknown', 'Unknown', 'Unknown',
                    'Inconnu', 'Inconnu', 'Inconnu')
            ON CONFLICT DO NOTHING
        """)
        self.conn.commit()
        logger.info("Inserted UNKNOWN food group")

    def insert_food_groups(self, food_groups: List[FoodGroup]) -> None:
        """
        Insert valid food groups into the clean food_groups table.

        Duplicate groups (based on the composite primary key) are ignored.

        Args:
            food_groups (List[FoodGroup]): List of FoodGroup objects to insert.
        """
        
        if not food_groups:
            return
        data = [(fg.alim_grp_code, fg.alim_ssgrp_code, fg.alim_ssssgrp_code,
                 fg.alim_grp_nom_eng, fg.alim_ssgrp_nom_eng, fg.alim_ssssgrp_nom_eng,
                 fg.alim_grp_nom_fr, fg.alim_ssgrp_nom_fr, fg.alim_ssssgrp_nom_fr)
                for fg in food_groups]
        execute_values(self.cur, """
            INSERT INTO food_groups
            (alim_grp_code, alim_ssgrp_code, alim_ssssgrp_code,
             alim_grp_nom_eng, alim_ssgrp_nom_eng, alim_ssssgrp_nom_eng,
             alim_grp_nom_fr, alim_ssgrp_nom_fr, alim_ssssgrp_nom_fr)
            VALUES %s
            ON CONFLICT (alim_grp_code, alim_ssgrp_code, alim_ssssgrp_code) DO NOTHING
        """, data)
        self.conn.commit()
        logger.info(f"Inserted {len(food_groups)} food groups into clean table")

    def insert_foods(self, food_groups: List[FoodGroup], foods: List[Food]) -> List[Food]:
        """
        Validate foods against existing food groups (including UNKNOWN), normalize
        placeholder group codes, insert valid foods, and log orphan foods.

        The function modifies the input Food objects in place (normalizing codes).

        Args:
            food_groups (List[FoodGroup]): List of valid food groups used for validation.
            foods (List[Food]): List of Food objects to validate and insert.

        Returns:
            List[Food]: The subset of input foods that were successfully inserted
            (valid groups). Orphan foods are logged into the orphan_foods table.
        """
        
        if not foods:
            return []

        valid_groups: Set[Tuple[str, str, str]] = {
            (fg.alim_grp_code, fg.alim_ssgrp_code, fg.alim_ssssgrp_code)
            for fg in food_groups
        }
        valid_groups.add(('UNKNOWN', 'UNKNOWN', 'UNKNOWN'))

        valid_foods = []
        orphan_foods = []

        for food in foods:
            grp = 'UNKNOWN' if food.alim_grp_code in ('00', '') else food.alim_grp_code
            ssgrp = 'UNKNOWN' if food.alim_ssgrp_code in ('0000', '') else food.alim_ssgrp_code
            ssssgrp = 'UNKNOWN' if food.alim_ssssgrp_code in ('000000', '') else food.alim_ssssgrp_code

            key = (grp, ssgrp, ssssgrp)
            if key in valid_groups:
                food.alim_grp_code = grp
                food.alim_ssgrp_code = ssgrp
                food.alim_ssssgrp_code = ssssgrp
                valid_foods.append(food)
            else:
                orphan_foods.append(food)

        if orphan_foods:
            logger.warning(f"Found {len(orphan_foods)} foods with missing group references")
            for food in orphan_foods:
                self.cur.execute("""
                    INSERT INTO orphan_foods (alim_code, reason)
                    VALUES (%s, %s)
                """, (food.alim_code, f"Missing group ({food.alim_grp_code}, {food.alim_ssgrp_code}, {food.alim_ssssgrp_code})"))

        if valid_foods:
            data = [(f.alim_code, f.alim_nom_eng, f.alim_nom_fr, f.alim_nom_sci,
                     f.facteur_jones, f.alim_grp_code, f.alim_ssgrp_code, f.alim_ssssgrp_code)
                    for f in valid_foods]
            execute_values(self.cur, """
                INSERT INTO foods
                (alim_code, alim_nom_eng, alim_nom_fr, alim_nom_sci,
                 facteur_jones, alim_grp_code, alim_ssgrp_code, alim_ssssgrp_code)
                VALUES %s
                ON CONFLICT (alim_code) DO UPDATE SET
                    alim_nom_eng = EXCLUDED.alim_nom_eng,
                    alim_nom_fr = EXCLUDED.alim_nom_fr,
                    alim_nom_sci = EXCLUDED.alim_nom_sci,
                    facteur_jones = EXCLUDED.facteur_jones,
                    alim_grp_code = EXCLUDED.alim_grp_code,
                    alim_ssgrp_code = EXCLUDED.alim_ssgrp_code,
                    alim_ssssgrp_code = EXCLUDED.alim_ssssgrp_code
            """, data)
            self.conn.commit()
            logger.info(f"Inserted/Updated {len(valid_foods)} foods into clean table")

        return valid_foods

    def insert_components(self, components: List[Component]) -> None:
        """
        Insert component data into the clean components table.

        Duplicate const_code entries are ignored.

        Args:
            components (List[Component]): List of Component objects to insert.
        """
        
        if not components:
            return
        data = [(c.const_code, c.const_nom_eng, c.const_nom_fr, c.code_infoods) for c in components]
        execute_values(self.cur, """
            INSERT INTO components (const_code, const_nom_eng, const_nom_fr, code_infoods)
            VALUES %s ON CONFLICT (const_code) DO NOTHING
        """, data)
        self.conn.commit()
        logger.info(f"Inserted {len(components)} components")

    def insert_data_sources(self, sources: List[DataSource]) -> None:
        """
        Insert data source references into the clean data_sources table.

        Duplicate source_code entries are ignored.

        Args:
            sources (List[DataSource]): List of DataSource objects to insert.
        """
        
        if not sources:
            return
        data = [(s.source_code, s.ref_citation) for s in sources]
        execute_values(self.cur, """
            INSERT INTO data_sources (source_code, ref_citation)
            VALUES %s ON CONFLICT (source_code) DO NOTHING
        """, data)
        self.conn.commit()
        logger.info(f"Inserted {len(sources)} data sources into clean table")

    def insert_composition(self, valid_foods: List[Food], compositions: List[Composition]) -> None:
        """
        Insert composition rows only for foods that were successfully loaded.

        Composition rows referencing missing foods are logged as orphans.

        Args:
            valid_foods (List[Food]): List of Food objects that exist in the foods table.
            compositions (List[Composition]): List of all Composition objects to filter.
        """
        
        if not compositions or not valid_foods:
            return

        food_codes = {f.alim_code for f in valid_foods}
        clean = []
        orphan = []

        for comp in compositions:
            if comp.alim_code in food_codes:
                clean.append(comp)
            else:
                orphan.append(comp)

        if orphan:
            logger.warning(f"Found {len(orphan)} composition rows with missing food references")
            for comp in orphan:
                self.cur.execute("""
                    INSERT INTO orphan_compositions (alim_code, const_code, reason)
                    VALUES (%s, %s, %s)
                """, (comp.alim_code, comp.const_code, f"Missing food code {comp.alim_code} in foods table"))

        if clean:
            data = [(c.alim_code, c.const_code, c.teneur, c.min_val,
                     c.max_val, c.code_confiance, c.source_code) for c in clean]
            execute_values(self.cur, """
                INSERT INTO composition
                (alim_code, const_code, teneur, min_val, max_val, code_confiance, source_code)
                VALUES %s
            """, data)
            self.conn.commit()
            logger.info(f"Inserted {len(clean)} composition records into clean table")

    def generate_reconciliation_report(self, output_dir: str = "reports"):
        """
        Generate a data quality report with statistics, charts, and CSV exports.

        The report includes:
            - Counts of foods and composition rows in staging, clean, and orphan tables.
            - Rejection rates.
            - Top reasons for orphan records.
            - Pie charts for valid vs orphan data.
            - Bar charts for top orphan reasons.
            - CSV exports of orphan records for further analysis.

        Args:
            output_dir (str, optional): Directory where charts and CSV files will be saved.
                Defaults to "reports". The directory is created if it does not exist.
        """
        
        os.makedirs(output_dir, exist_ok=True)

        # 1. Food statistics
        self.cur.execute("SELECT COUNT(*) FROM staging_foods")
        staging_foods = self.cur.fetchone()[0]
        self.cur.execute("SELECT COUNT(*) FROM foods")
        clean_foods = self.cur.fetchone()[0]
        self.cur.execute("SELECT COUNT(*) FROM orphan_foods")
        orphan_foods = self.cur.fetchone()[0]

        # 2. Composition statistics
        self.cur.execute("SELECT COUNT(*) FROM staging_composition")
        staging_compo = self.cur.fetchone()[0]
        self.cur.execute("SELECT COUNT(*) FROM composition")
        clean_compo = self.cur.fetchone()[0]
        self.cur.execute("SELECT COUNT(*) FROM orphan_compositions")
        orphan_compo = self.cur.fetchone()[0]

        # 3. Orphan reason analysis (from orphan_foods table)
        self.cur.execute("SELECT reason FROM orphan_foods")
        reason_counts = Counter(row[0] for row in self.cur.fetchall())
        
        # 4. Orphan composition reasons (from orphan_compositions table)
        self.cur.execute("SELECT reason FROM orphan_compositions")
        compo_reason_counts = Counter(row[0] for row in self.cur.fetchall())

        # Print summary
        print("\n" + "="*60)
        print("RECONCILIATION REPORT")
        print("="*60)
        print(f"Foods:\n  - Source (staging): {staging_foods:,}\n  - Loaded (clean): {clean_foods:,}\n  - Orphan (logged): {orphan_foods:,}\n  - Rejected rate: {(staging_foods - clean_foods)/staging_foods*100:.1f}%")
        print(f"\nComposition rows:\n  - Source (staging): {staging_compo:,}\n  - Loaded (clean): {clean_compo:,}\n  - Orphan (logged): {orphan_compo:,}\n  - Rejected rate: {(staging_compo - clean_compo)/staging_compo*100:.1f}%")
        if reason_counts:
            print("\nTop 5 reasons for orphan foods:")
            for r, c in reason_counts.most_common(5):
                print(f"  - {r[:80]}: {c}")
        if compo_reason_counts:
            print("\nTop 5 reasons for orphan compositions:")
            for r, c in compo_reason_counts.most_common(5):
                print(f"  - {r[:80]}: {c}")

        # 2*2 chart
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("CIQUAL ETL Data Quality Report", fontsize=16)
        
        # Pie: Foods
        axes[0,0].pie([clean_foods, orphan_foods], labels=['Valid Foods', 'Orphan Foods'],
                      autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
        axes[0,0].set_title(f"Foods (Total source: {staging_foods:,})")
        # Pie: Composition
        axes[0,1].pie([clean_compo, orphan_compo], labels=['Valid Composition', 'Orphan Composition'],
                      autopct='%1.1f%%', colors=['#3498db', '#e67e22'], startangle=90)
        axes[0,1].set_title(f"Composition rows (Total source: {staging_compo:,})")
        
        # Bar: Orphan food reasons (top 5)
        if reason_counts:
            reasons_top = list(reason_counts.keys())[:5]
            counts_top = [reason_counts[r] for r in reasons_top]
            short_reasons = [r[:40] + '...' if len(r) > 40 else r for r in reasons_top]
            sns.barplot(data=pd.DataFrame({'reason': short_reasons, 'count': counts_top}),
                        x='count', y='reason', hue='reason', ax=axes[1,0], palette='Reds_d', legend=False)
            axes[1,0].set_title("Top Orphan Food Reasons")
            axes[1,0].set_xlabel("Number of foods")
        else:
            axes[1,0].text(0.5, 0.5, "No orphan foods", ha='center')

        # Bar: Orphan composition reasons (top 5)
        if compo_reason_counts:
            compo_top = list(compo_reason_counts.keys())[:5]
            compo_counts = [compo_reason_counts[r] for r in compo_top]
            short_cr = [r[:40] + '...' if len(r) > 40 else r for r in compo_top]
            sns.barplot(data=pd.DataFrame({'reason': short_cr, 'count': compo_counts}),
                        x='count', y='reason', hue='reason', ax=axes[1,1], palette='Oranges_d', legend=False)
            axes[1,1].set_title("Top Orphan Composition Reasons")
            axes[1,1].set_xlabel("Number of composition rows")
        else:
            axes[1,1].text(0.5, 0.5, "No orphan compositions", ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "reconciliation_charts.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nCharts saved to: {os.path.join(output_dir, 'reconciliation_charts.png')}")

        # Export CSV
        try:
            self.cur.execute("SELECT alim_code, reason, created_at FROM orphan_foods")
            orphan_foods_rows = self.cur.fetchall()
            if orphan_foods_rows:
                pd.DataFrame(orphan_foods_rows, columns=['alim_code', 'reason', 'created_at']).to_csv(
                    os.path.join(output_dir, "orphan_foods.csv"), index=False)
            self.cur.execute("SELECT alim_code, const_code, reason, created_at FROM orphan_compositions")
            orphan_compo_rows = self.cur.fetchall()
            if orphan_compo_rows:
                pd.DataFrame(orphan_compo_rows, columns=['alim_code', 'const_code', 'reason', 'created_at']).to_csv(
                    os.path.join(output_dir, "orphan_compositions.csv"), index=False)
        except Exception as e:
            print(f"Warning: Could not export orphan CSVs: {e}")