import streamlit as st
import uuid

st.set_page_config(page_title="💼 Sales Bonus System", layout="centered")
st.title("🏆 Sales Incentive Calculator")

# Internal bonus slabs (hidden from user)
bonus_slabs = [
    {"min": 0.0, "max": 5.0, "bonus": 0},
    {"min": 5.0, "max": 10.0, "bonus": 10},
    {"min": 10.0, "max": 15.0, "bonus": 12},
    {"min": 15.0, "max": 999.0, "bonus": 15},
]

# Step 1: Ask name and ID
st.header("👤 Salesperson Details")
name = st.text_input("Enter your name:")
user_id = st.text_input("Enter your Employee ID (or leave blank to auto-generate):")

# Auto-generate ID
if name and not user_id:
    user_id = str(uuid.uuid4()).split("-")[0]

# Step 2: Sales entry
if name and user_id:
    st.success(f"Welcome, **{name}** (ID: `{user_id}`)")
    st.divider()

    st.header("📈 Enter Your Sales")
    sales_input = st.text_input("Enter total sales (₹): Can be in lakh or full rupees (e.g., 10.5 or 1250000)")

    if sales_input:
        try:
            raw_sales = float(sales_input)

            # Detect if input is in rupees and convert to lakhs
            if raw_sales > 1000:  # simple threshold logic
                sales_lakh = raw_sales / 100000
                st.info(f"🧠 Detected input as rupees → converted to ₹{sales_lakh:.2f} lakh")
            else:
                sales_lakh = raw_sales

            # Match against slab
            matched_bonus = 0
            for slab in bonus_slabs:
                if slab["min"] <= sales_lakh < slab["max"]:
                    matched_bonus = slab["bonus"]
                    break

            bonus_amount = (matched_bonus / 100) * sales_lakh * 100000

            # Display result
            st.success(f"Your sales: ₹{sales_lakh:.2f} lakh")
            if matched_bonus > 0:
                st.balloons()
                st.markdown(f"""
                ### 🎉 Bonus Summary:
                - 👤 Name: **{name}**  
                - 🆔 ID: `{user_id}`  
                - 💼 Sales: ₹{sales_lakh:.2f} lakh  
                - 🎯 Bonus Rate: **{matched_bonus}%**
                - 💰 Total Bonus: **₹{bonus_amount:,.2f}**
                """)
            else:
                st.warning("No bonus applicable for this sales amount.")

        except ValueError:
            st.error("❌ Please enter a valid number.")
