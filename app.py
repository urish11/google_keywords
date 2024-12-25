if "all_data" in st.session_state:
    all_data = st.session_state["all_data"]

    cluster_data = dynamic_keyword_clustering(all_data["Keyword"].tolist(), ngram_range=(2, 3), eps=0.6, min_samples=2)
    aggregated_table, keywords_map = aggregate_by_cluster(all_data, cluster_data)

    # Debugging and Cleaning
    st.write("Data Types Before AgGrid:")
    st.write(aggregated_table.dtypes)
    aggregated_table.fillna("", inplace=True)
    aggregated_table = aggregated_table.astype(str)  # Ensure string conversion

    # Display Aggregated Table
    st.write("### Aggregated Table (By Key Phrase)")
    gb = GridOptionsBuilder.from_dataframe(aggregated_table)
    gb.configure_pagination(enabled=True, paginationPageSize=100)
    gb.configure_default_column(filterable=True, sortable=True, editable=False)
    gb.configure_column("Cluster Keywords", cellStyle={"cursor": "pointer"})
    grid_options = gb.build()

    try:
        grid_response = AgGrid(aggregated_table, gridOptions=grid_options, height=800, width=700, theme="streamlit")
    except ValueError as e:
        st.error(f"Error rendering AgGrid: {e}")
        st.write("Problematic DataFrame:")
        st.write(aggregated_table)
